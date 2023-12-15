// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on

#include <torch/extension.h>

namespace {

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
void paged_attention_v1_xpu_impl(
    scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int *__restrict__ head_mapping, // [num_heads]
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const int num_seqs, const int num_heads, const int num_blocks) {
  TORCH_CHECK(HEAD_SIZE % 16 == 0);
  TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
  constexpr int x = 16 / sizeof(scalar_t);

  int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
  int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
  TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

  sycl::queue task_q(sycl::gpu_selector_v);
  sycl::buffer<scalar_t, 1> out_buf(out, num_seqs * num_heads * HEAD_SIZE);
  sycl::buffer<scalar_t, 1> q_buf(q, num_seqs * num_heads * HEAD_SIZE);
  sycl::buffer<int, 1> head_mapping_buf(head_mapping, num_heads);
  sycl::buffer<int, 1> context_lens_buf(context_lens, num_seqs);
  sycl::buffer<int, 1> block_tables_buf(block_tables, num_seqs * max_num_blocks_per_seq);
  const int new_kv_block_stride = kv_block_stride;

  sycl::buffer<scalar_t, 1> k_cache_buf(k_cache, num_blocks * kv_block_stride); 
  sycl::buffer<scalar_t, 1> v_cache_buf(v_cache, num_blocks * kv_block_stride); 
  size_t logits_bytes = num_heads * max_context_len_padded * sizeof(float);
  float *logits = (float *)sycl::aligned_alloc_device(
      64, logits_bytes, task_q.get_device(),
      task_q.get_context()); // Cacheline alignment for each context token.
                             // [head_num, max_context_len_padded]
  std::memset(out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));
  // sycl::event reset_output = task_q.memset(
  //     out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));
  constexpr int head_partition_num = HEAD_SIZE / 16;
  
  for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    int context_len = context_lens[seq_idx];
    const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::event reset_logits = task_q.memset(logits, 0, logits_bytes);
    reset_logits.wait();

    auto e1 = task_q.submit([&](auto &h)
                            {
      sycl::accessor q_acc(q_buf, h, sycl::read_only);
      sycl::accessor k_cache_acc(k_cache_buf, h, sycl::read_only);
      sycl::accessor head_mapping_acc(head_mapping_buf, h, sycl::read_only);
      sycl::accessor context_lens_acc(context_lens_buf, h, sycl::read_only);
      sycl::accessor block_tables_acc(block_tables_buf, h, sycl::read_only);
  
      h.parallel_for(
        sycl::range(num_heads,  HEAD_SIZE/x, x),
        [=](sycl::item<3> item) {
          size_t head_idx = item[0];
          size_t x_idx = item[1];
          size_t i = item[2];
          const int32_t kv_head_idx = head_mapping_acc[head_idx];
          size_t q_base_offset = seq_idx * q_stride + head_idx * HEAD_SIZE; // dim0, dim1
          
          for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
            const int32_t physical_block_idx = block_tables_acc[block_idx + max_num_blocks_per_seq * seq_idx];
            size_t k_base_offset = physical_block_idx * kv_block_stride +
                                   kv_head_idx * kv_head_stride; // dim0,dim1
            float *__restrict__ head_block_logits =
                logits + head_idx * max_context_len_padded + block_idx * BLOCK_SIZE;
            for (int token_idx = 0; token_idx < BLOCK_SIZE; ++token_idx) {
              head_block_logits[token_idx] += q_acc[i + x_idx * x + q_base_offset] *
              k_cache_acc[i + token_idx * x + BLOCK_SIZE * x_idx * x + k_base_offset] * scale;
            }    
          }            
      });
    });
    e1.wait();

    auto e2 = task_q.submit([&](auto &h) {
          h.parallel_for(sycl::range(num_heads), [=](sycl::item<1> item) {
          size_t head_idx = item[0];
          float *head_logit_ptr = logits + head_idx * max_context_len_padded;
          float max_logit = head_logit_ptr[0];
          for (int i = 1; i < context_len; ++i) {
            max_logit =
                max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
          }
          float sum = 0;
          for (int i = 0; i < context_len; ++i) {
            head_logit_ptr[i] = sycl::exp(head_logit_ptr[i] - max_logit);
            sum += head_logit_ptr[i];
          }
          for (int i = 0; i < context_len; ++i) {
            head_logit_ptr[i] /= sum;
          }
          int remaining_seq_upper = block_num * BLOCK_SIZE;
          for (int i = context_len; i < remaining_seq_upper; ++i) {
            head_logit_ptr[i] = 0;
          } }); });

    e2.wait();

    auto e3 = task_q.submit([&](auto &h) {
      sycl::accessor output_acc(out_buf, h, sycl::read_write);
      sycl::accessor v_cache_acc(v_cache_buf, h, sycl::read_only);
      sycl::accessor head_mapping_acc(head_mapping_buf, h, sycl::read_only);
      sycl::accessor context_lens_acc(context_lens_buf, h, sycl::read_only);
      sycl::accessor block_tables_acc(block_tables_buf, h, sycl::read_only);

      h.parallel_for(
        sycl::range(num_heads, head_partition_num, block_num),
        [=](sycl::item<3> item) {
          size_t head_idx = item[0];
          size_t head_part_idx = item[1];
          size_t block_idx = item[2];

          // for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int32_t kv_head_idx = head_mapping_acc[head_idx];
            const int32_t physical_block_idx = block_tables_acc[block_idx + max_num_blocks_per_seq * seq_idx];
            const scalar_t *__restrict__ prob_vec_ptr =
                logits + head_idx * max_context_len_padded +
                block_idx * BLOCK_SIZE;
            size_t v_base_offset = physical_block_idx * new_kv_block_stride +
                                   kv_head_idx * kv_head_stride + BLOCK_SIZE * head_part_idx * 16;
            size_t out_base_offset = seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_part_idx * 16;
            for (int i = 0; i < 16; ++i) {
              for (int j = 0; j < BLOCK_SIZE; ++j) {
                output_acc[i + out_base_offset] += 
                prob_vec_ptr[j] * 
                v_cache_acc[j + i * BLOCK_SIZE + v_base_offset];
              }
            }
          // }
        });        
    });

    e3.wait();

  }
  sycl::free(logits, task_q);
}

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                      \
  paged_attention_v1_xpu_impl<T, HEAD_SIZE, BLOCK_SIZE>(                       \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, head_mapping_ptr,    \
      scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,       \
      alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads, num_blocks);

template <typename T, int BLOCK_SIZE>
void paged_attention_v1_xpu_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);
  int num_blocks = key_cache.size(0);
  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *head_mapping_ptr = reinterpret_cast<int *>(head_mapping.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
  // 32, 160, 192.
  // case 32:
  //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
  //   break;
  case 64:
    LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  // case 160:
  //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE);
  //   break;
  // case 192:
  //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE);
  //   break;
  case 256:
    LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                    \
  paged_attention_v1_xpu_launcher<T, BLOCK_SIZE>(                              \
      out, query, key_cache, value_cache, head_mapping, scale, block_tables,   \
      context_lens, max_context_len, alibi_slopes);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                     \
  switch (block_size) {                                                        \
  /* case 1:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 1);   */                                        \
  /*   break;                        */                                        \
  /* case 2:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 2);   */                                        \
  /*   break;                        */                                        \
  /* case 4:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 4);   */                                        \
  /*   break;                        */                                        \
  case 8:                                                                      \
    CALL_KERNEL_LAUNCHER(T, 8);                                                \
    break;                                                                     \
  case 16:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 16);                                               \
    break;                                                                     \
  case 32:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 32);                                               \
    break;                                                                     \
  /* case 64:                        */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 64);  */                                        \
  /*   break;                        */                                        \
  /* case 128:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 128); */                                        \
  /*   break;                        */                                        \
  /* case 256:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 256); */                                        \
  /*   break;                        */                                        \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }
} // namespace

void paged_attention_v1_xpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, torch::Tensor &head_mapping, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

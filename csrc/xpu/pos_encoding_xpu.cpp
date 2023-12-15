// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on

#include <torch/extension.h>

template <typename scalar_t>
void rotary_embedding_xpu_impl(
    const int64_t *__restrict__ positions, // [num_tokens]
    scalar_t *__restrict__ query,          // [num_tokens, num_heads, head_size]
    scalar_t *__restrict__ key, // [num_tokens, num_kv_heads, head_size]
    const scalar_t *__restrict__ cos_sin_cache, // [max_position, 2, rot_dim //
                                                // 2]
    const int rot_dim, const int query_stride, const int key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens, const int sin_cos_dim) {
  const int embed_dim = rot_dim / 2;
  sycl::buffer<int64_t, 1> positions_buf(positions, num_tokens);
  sycl::buffer<scalar_t, 3> query_buf(query, sycl::range<3>(num_tokens, num_heads, head_size));
  sycl::buffer<scalar_t, 3> key_buf(key, sycl::range<3>(num_tokens, num_kv_heads, head_size));
  sycl::buffer<scalar_t, 1> cos_sin_cache_buf(cos_sin_cache, sin_cos_dim * rot_dim); 
  sycl::queue q(sycl::gpu_selector_v);
  q.submit([&](auto &h) {
    sycl::accessor positions_acc(positions_buf, h, sycl::read_only);
    sycl::accessor query_acc(query_buf, h, sycl::read_write);
    sycl::accessor key_acc(key_buf, h, sycl::read_write);
    sycl::accessor cos_sin_cache_acc(cos_sin_cache_buf, h, sycl::read_only);

    h.parallel_for(sycl::range(num_tokens, num_heads, embed_dim), [=](auto index) {
      const long token_idx = index[0];
      const long head_idx = index[1];
      const int x_index = index[2];
      const int y_index = x_index + embed_dim;
      int64_t pos = positions_acc[token_idx];
      
      const scalar_t q_x = query_acc[token_idx][head_idx][x_index];
      const scalar_t q_y = query_acc[token_idx][head_idx][y_index];
      const scalar_t cos = cos_sin_cache_acc[pos * rot_dim + x_index];
      const scalar_t sin = cos_sin_cache_acc[pos * rot_dim + y_index];

      query_acc[token_idx][head_idx][x_index] = q_x * cos - q_y * sin;
      query_acc[token_idx][head_idx][y_index] = q_y * cos + q_x * sin;

      if (head_idx < num_kv_heads) {
        const scalar_t k_x = key_acc[token_idx][head_idx][x_index];
        const scalar_t k_y = key_acc[token_idx][head_idx][y_index];
        key_acc[token_idx][head_idx][x_index] = k_x * cos - k_y * sin;
        key_acc[token_idx][head_idx][y_index] = k_y * cos + k_x * sin;
      }
    });
  });

  q.wait();
}

void _rotary_embedding_xpu(torch::Tensor &positions, torch::Tensor &query,
                           torch::Tensor &key, int head_size,
                           torch::Tensor &cos_sin_cache) {
  TORCH_CHECK(query.scalar_type() == c10::ScalarType::Float);

  int num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int key_stride = key.stride(-2);
  int query_stride = query.stride(-2);
  int cos_sin_dim = cos_sin_cache.size(0);

  AT_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "rotary_embedding_xpu_impl", [&] {
        rotary_embedding_xpu_impl(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
            head_size, num_tokens, cos_sin_dim);
      });
}

void rotary_embedding_xpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox) {
  TORCH_CHECK(is_neox);
  switch (positions.device().type()) {
  case c10::DeviceType::CPU:
    return _rotary_embedding_xpu(positions, query, key, head_size,
                                 cos_sin_cache);
  default:
    TORCH_CHECK(false, "Unsupported device type.")
  }
}

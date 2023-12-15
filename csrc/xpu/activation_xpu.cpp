// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on

#include <torch/extension.h>

template <typename T> __inline__ T silu_xpu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename scalar_t>
void silu_and_mul_xpu_impl(int num_tokens, int d,
                           const scalar_t *__restrict__ input, //
                           scalar_t *__restrict__ output) {
  sycl::queue q{sycl::gpu_selector_v};
  sycl::buffer<scalar_t, 1> input_buf(input, num_tokens * d * 2);
  sycl::buffer<scalar_t, 1> output_buf(output, num_tokens * d);
  q.submit([&](auto &h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor output_acc(output_buf, h, sycl::read_write);
  
    // each work item calculate 16 output result, trying to leverage SIMD lane
    h.parallel_for(sycl::nd_range<1>(num_tokens * d / 16, 128), [=](sycl::nd_item<1>index) {
      int i = index.get_global_linear_id();
      auto sg = index.get_sub_group();
      int sgSize = sg.get_local_range()[0];
      i = (i / sgSize) * sgSize * 16 + (i % sgSize);

      for (int j=0; j< 16 * sgSize; j += sgSize){
        const int idx = i + j;
        const int token_idx = idx / d;
        const int dim_idx = idx % d; 
        const scalar_t x = input_acc[token_idx * d * 2 + dim_idx];
        const scalar_t y = input_acc[token_idx * d * 2 + dim_idx + d];
        output_acc[i + j] =  silu_xpu(x) * y;
      }
    });

  });
  q.wait();
}

void silu_and_mul_xpu(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float);
  int num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_xpu_impl", [&] {
    silu_and_mul_xpu_impl(num_tokens, d, input.data_ptr<scalar_t>(),
                          out.data_ptr<scalar_t>());
  });
}

// void gelu_new(torch::Tensor &out, torch::Tensor &input);

// void gelu_fast(torch::Tensor &out, torch::Tensor &input);

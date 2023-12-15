// clang-format off
#ifdef VLLM_DEV
#undef __SYCL_DEVICE_ONLY__
#endif
#include <sycl/sycl.hpp>
// clang-format on

#include <torch/extension.h>
#include <algorithm>

template <typename scalar_t>
void rms_norm_xpu_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const size_t num_tokens,
    const size_t hidden_size) {
  sycl::queue q(sycl::gpu_selector_v);
  sycl::buffer<scalar_t, 2> input_buf(
      input, sycl::range(num_tokens, hidden_size));
  sycl::buffer<scalar_t, 1> weight_buf(weight, hidden_size);
  sycl::buffer<scalar_t, 2> out_buf(out, sycl::range(num_tokens, hidden_size));

  sycl::range<2> global_size = sycl::range<2>{num_tokens, hidden_size};
  size_t size_1 = std::min(num_tokens, 32ul);
  size_t size_2 = std::min(hidden_size, 32ul);
  sycl::range<2> local_size = sycl::range<2>{size_1, size_2};
  size_t s = (hidden_size + size_2 - 1) / size_2;
  scalar_t* accum_data = sycl::malloc_device<scalar_t>(num_tokens * s, q);
  // scalar_t* accum_data = (scalar_t*)malloc(num_tokens * s * sizeof(scalar_t));

  sycl::buffer<scalar_t, 2> accum_buf(
      accum_data, sycl::range<2>{num_tokens, s});

  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);
    sycl::local_accessor<scalar_t, 2> scratch(local_size, h);

    h.parallel_for(
        sycl::nd_range<2>(global_size, local_size),
        [=](sycl::nd_item<2> index) {
          size_t g_row_id = index.get_global_id()[0];
          size_t g_col_id = index.get_global_id()[1];
          size_t l_row_id = index.get_local_id()[0];
          size_t l_col_id = index.get_local_id()[1];
          int group_col_id = index.get_group(1);
          scalar_t x = input_acc[g_row_id][g_col_id];
          scratch[l_row_id][l_col_id] = x * x;
          for (int i = size_2 / 2; i > 0; i >>= 1) {
            index.barrier(sycl::access::fence_space::local_space);
            if (l_col_id < i && (g_col_id + i < hidden_size)) {
              scratch[l_row_id][l_col_id] += scratch[l_row_id][l_col_id + i];
            }
          }
          if (l_col_id == 0) {
            accum_acc[g_row_id][group_col_id] = scratch[l_row_id][0];
          }
        });
  });
  q.wait();

  q.submit([&](auto& h) {
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

    h.parallel_for(num_tokens, [=](auto index) {
      size_t row_id = index[0];
      for (int i = 1; i < s; ++i) {
        accum_acc[row_id][0] += accum_acc[row_id][i];
      }
      accum_acc[row_id][0] =
          sycl::rsqrt(accum_acc[row_id][0] / (float)hidden_size + epsilon);
    });
  });
  q.wait();
  q.submit([&](auto& h) {
    sycl::accessor input_acc(input_buf, h, sycl::read_only);
    sycl::accessor weight_acc(weight_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::read_write);
    sycl::accessor out_acc(out_buf, h, sycl::read_write);

    h.parallel_for(
        sycl::nd_range<2>(global_size, local_size),
        [=](sycl::nd_item<2> index) {
          size_t row_id = index.get_global_id()[0];
          size_t col_id = index.get_global_id()[1];
          scalar_t x = input_acc[row_id][col_id];
          out_acc[row_id][col_id] =
              ((scalar_t)(x * accum_acc[row_id][0])) * weight_acc[col_id];
        });
  });
  q.wait();
  sycl::free(accum_data, q);
  // free(accum_data);
  // q.wait();
}

void _rms_norm_xpu(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_xpu_impl", [&] {
    rms_norm_xpu_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
  });
}

void rms_norm_xpu(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  switch (weight.device().type()) {
    case c10::DeviceType::CPU:
      return _rms_norm_xpu(out, input, weight, epsilon);
    default:
      TORCH_CHECK(false, "Unsupported device type.")
  }
}

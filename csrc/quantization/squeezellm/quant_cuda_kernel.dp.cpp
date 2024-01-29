#include <torch/all.h>
#include <torch/python.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>


#define BLOCKWIDTH 128
#define BLOCKHEIGHT4 16

namespace vllm {
namespace squeezellm {

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// 4-bit matvec kernel (LUT-based)
void NUQ4MatMulKernel(
    sycl::half2 * __restrict__  vec,
    int * __restrict__  mat,
    sycl::half2 * __restrict__  mul,
    sycl::half * __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height,
    const sycl::nd_item<3> &item_ct1,
    sycl::accessor<sycl::half2, 1> blockvec,
    sycl::accessor<sycl::half, 2> deq2
    ) {

  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  sycl::half res;
#ifndef USE_ROCM
  sycl::half2 res2;
  sycl::half2 tmp2;
#else
  __half2 res2;
  __half2 tmp2;
#endif

  int i;
  int k;

  unsigned int tmp1;
  unsigned int lut_index1, lut_index2;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res =
        sycl::vec<int, 1>{0}.convert<sycl::half, sycl::rounding_mode::rtn>()[0];
    k = 0;

    /*
    DPCT1118:0: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < blockwidth2)
      blockvec[item_ct1.get_local_id(2)] =
          vec[b * vec_height / 2 + (row / BLOCKHEIGHT4) * blockwidth2 +
              item_ct1.get_local_id(2)];
    /*
    DPCT1118:1: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    while (k < blockwidth2) {
      tmp1 = as_unsigned(mat[i]);

#ifndef USE_ROCM
      res2 = {};
      tmp2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
      tmp2.x = __half_as_ushort(__float2half(0));
      tmp2.y = __half_as_ushort(__float2half(0));
#endif

      lut_index1 = tmp1 & 0xF;
      lut_index2 = (tmp1 >> 4) & 0xF;
#ifndef USE_ROCM
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
      res2 = sycl::fma(tmp2, blockvec[k + 0], res2);

      lut_index1 = (tmp1 >> 8) & 0xF;
      lut_index2 = (tmp1 >> 12) & 0xF;
#ifndef USE_ROCM
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
      res2 = sycl::fma(tmp2, blockvec[k + 1], res2);

      lut_index1 = (tmp1 >> 16) & 0xF;
      lut_index2 = (tmp1 >> 20) & 0xF;
#ifndef USE_ROCM
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
      res2 = sycl::fma(tmp2, blockvec[k + 2], res2);

      lut_index1 = (tmp1 >> 24) & 0xF;
      lut_index2 = (tmp1 >> 28) & 0xF;
#ifndef USE_ROCM
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
#else
      tmp2.x = __half_as_ushort(deq2[lut_index1][off]);
      tmp2.y = __half_as_ushort(deq2[lut_index2][off]);
#endif
      res2 = sycl::fma(tmp2, blockvec[k + 3], res2);

#ifndef USE_ROCM
      res = res2.x() + res2.y() + res;
#else
      res = __hadd(__hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)), res);
#endif

      i += width;
      k += 4;
    }

    // col%2 -> only set one of the two values
#ifndef USE_ROCM
    sycl::half2 res3 = {};
    if (col % 2 == 0) {
      res3.x() = res;
    } else {
      res3.y() = res;
    }
#else
    __half2 res3;
    res3.x = __half_as_ushort(__float2half(0));
    res3.y = __half_as_ushort(__float2half(0));
     if (col % 2 == 0) {
      res3.x = __half_as_ushort(res);
    } else {
      res3.y = __half_as_ushort(res);
    }
#endif

// #ifndef USE_ROCM
//     /*
//     DPCT1007:4: Migration of half version of atomicAdd is not supported.
//     */
//     atomicAdd(&mul[b * width / 2 + col / 2], res3);
// #else
//     int tmp_addr = b * width / 2 + col / 2;
//     atomicAdd(&(mul[tmp_addr].x), __half2float(__ushort_as_half(res3.x)));
//     atomicAdd(&(mul[tmp_addr].y), __half2float(__ushort_as_half(res3.y)));
// #endif
  }
}

} // namespace squeezellm
} // namespace vllm

// 4-bit matvec kernel (LUT-based)
void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  // poiter conversions
  sycl::half2* vec_ptr = static_cast<sycl::half2*>(vec.data_ptr());
  int* int_ptr = static_cast<int*>(mat.data_ptr());
  sycl::half2* mul_ptr = static_cast<sycl::half2*>(mul.data_ptr());
  sycl::half* lookup_table_ptr = static_cast<sycl::half*>(lookup_table.data_ptr());
  
  // buffer
  sycl::buffer<sycl::half2, 1> blockvec{sycl::range<1>(BLOCKWIDTH / 2)};
  sycl::buffer<sycl::half, 2> deq2{sycl::range<2>(16, BLOCKWIDTH)};

  // Create a queue
  sycl::queue q;
  // Submit a command group to the queue
  q.submit([&](sycl::handler& cgh) {
  // <<<blocks, threads>>> --> (global, local)
  // global work size
  sycl::range<3> global_size(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH, (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  // local work-group size
  sycl::range<3> local_size(1, 1, BLOCKWIDTH);
      
  sycl::accessor blockvec_acc(blockvec, cgh, sycl::read_write);
  sycl::accessor deq2_acc(deq2, cgh, sycl::read_write);
      //Set up the kernel
      cgh.parallel_for(sycl::nd_range<3>(global_size, local_size), [=](sycl::nd_item<3> item) {
            // Call the kernel function
            vllm::squeezellm::NUQ4MatMulKernel(
              vec_ptr,
              int_ptr,
              mul_ptr,
              lookup_table_ptr,
              height, width, batch, vec_height,
              item,
              blockvec_acc,deq2_acc
            );
      });
  });

}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT4
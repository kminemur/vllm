/*
Adapted from https://github.com/turboderp/exllamav2 and https://github.com/turboderp/exllama
*/

#ifndef _matrix_view_cuh
#define _matrix_view_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "qdq_util.dp.hpp"

namespace vllm {
namespace gptq {

class MatrixView_half
{
public:
    const sycl::half *data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_half(const sycl::half *data, const int height,
                                    const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ sycl::half item(int row, int column) const {
        return data[row * width + column];
    }
    __dpct_inline__ sycl::half2 item_half2(int row, int column) const {
        return ((sycl::half2 *)data)[(row * width + column) / 2];
    }
    __dpct_inline__ sycl::half2 item_half2half2(int row, int column) const {
        return sycl::half2{data[row * width + column],
                           data[row * width + column]};
    }
    __dpct_inline__ const sycl::half *item_ptr(int row, int column) const {
        return &data[row * width + column];
    }

    __dpct_inline__ void item4(sycl::half (&items)[4], int row,
                               int column) const
    {
        sycl::half2 *ptr = (sycl::half2 *)item_ptr(row, column);
        sycl::half2 i01 = ptr[0];
        sycl::half2 i23 = ptr[1];
        items[0] = i01[1];
        items[1] = i01[0];
        items[2] = i23[1];
        items[3] = i23[0];
    }
    __dpct_inline__ void item4_f(float (&items)[4], int row, int column) const
    {
        sycl::half2 *ptr = (sycl::half2 *)item_ptr(row, column);
        sycl::half2 i01 = ptr[0];
        sycl::half2 i23 = ptr[1];
        items[0] = sycl::vec<sycl::half, 1>{i01[1]}
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        items[1] = sycl::vec<sycl::half, 1>{i01[0]}
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        items[2] = sycl::vec<sycl::half, 1>{i23[1]}
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        items[3] = sycl::vec<sycl::half, 1>{i23[0]}
                       .convert<float, sycl::rounding_mode::automatic>()[0];
    }

    __dpct_inline__ void item4_h2(sycl::half2 (&items)[4], int row,
                                  int column) const
    {
        sycl::half2 *ptr = (sycl::half2 *)item_ptr(row, column);
        sycl::half2 i01 = ptr[0];
        sycl::half2 i23 = ptr[1];
        items[0] = sycl::half2{i01[1], i01[1]};
        items[1] = sycl::half2{i01[0], i01[0]};
        items[2] = sycl::half2{i23[1], i23[1]};
        items[3] = sycl::half2{i23[0], i23[0]};
    }
};

class MatrixView_half_rw
{
public:
    sycl::half *data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_half_rw(sycl::half *data, const int height,
                                       const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ sycl::half item(int row, int column) const {
        return data[row * width + column];
    }
    __dpct_inline__ sycl::half2 item_half2(int row, int column) const {
        return ((sycl::half2 *)data)[(row * width + column) / 2];
    }
    __dpct_inline__ sycl::half *item_ptr(int row, int column) {
        return &data[row * width + column];
    }
    __dpct_inline__ void set(int row, int column, sycl::half value) {
        data[row * width + column] = value;
    }
    __dpct_inline__ void set_half2(int row, int column, sycl::half2 value) {
        ((sycl::half2 *)data)[(row * width + column) / 2] = value;
    }

    __dpct_inline__ void set4(int row, int column, sycl::half v0, sycl::half v1,
                              sycl::half v2, sycl::half v3)
    {
        sycl::half2 v01 = sycl::half2{v0, v1};
        sycl::half2 v23 = sycl::half2{v2, v3};
        sycl::half2 *ptr = (sycl::half2 *)item_ptr(row, column);
        ptr[0] = v01;
        ptr[1] = v23;
    }
};

class MatrixView_q4_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_q4_row(const uint32_t *data, const int height,
                                      const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ int item(int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        return (data[row * width / 8 + column / 8] >> shift) & 0x0f;
    }

    __dpct_inline__ void item2(int (&items)[2], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
    }

    __dpct_inline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
        items[2] = (d >> 8) & 0x0f;
        items[3] = (d >> 12) & 0x0f;
    }
};

class MatrixView_q4_column
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __dpct_inline__ MatrixView_q4_column(const uint32_t *data, const int height,
                                         const int width)
        : data(data), height(height), width(width)
    { }

    __dpct_inline__ int item(int row, int column) const
    {
        int shift = (row & 0x07) * 4;
        return (data[row / 8 * width + column] >> shift) & 0x0f;
    }

    __dpct_inline__ uint32_t item_uint32_t(int row, int column) {
        return data[row / 8 * width + column];
    }
    __dpct_inline__ const uint32_t *item_uint32_ptr(int row, int column) {
        return &data[row / 8 * width + column];
    }
};

}  // namespace gptq
}  // namespace vllm
#endif

#ifndef COO_MATMUL_H
#define COO_MATMUL_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <string>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <numeric>
// #include <ips4o.hpp>
#include <tbb/parallel_sort.h>

// Define a custom hash function for std::pair<int, int>
struct PairHash
{
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U> &p) const
    {
        auto h1 = std::hash<T>{}(p.first);
        auto h2 = std::hash<U>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Define a custom equality comparison function for std::pair<int, int>
struct PairEqual
{
    template <typename T, typename U>
    bool operator()(const std::pair<T, U> &lhs, const std::pair<T, U> &rhs) const
    {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

struct ArrayHash
{
    std::size_t operator()(const std::vector<int> &arr) const
    {
        std::size_t seed = 0;
        for (int val : arr)
        {
            seed ^= std::hash<int>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct Entry
{
    uint32_t offset;
    uint16_t num_idx;
    double value;
};

struct COOMatrix
{
    std::vector<std::vector<int>> data; // Integer indices
    std::vector<double> values;         // Values
    std::vector<int> shape;             // Shape of the matrix

    COOMatrix(const std::vector<std::vector<int>> &data, const std::vector<double> &values, const std::vector<int> &shape)
        : data(data), values(values), shape(shape) {}
};

void coo_bmm(const double *A_data, int A_rows, int A_cols,
             const double *B_data, int B_rows, int B_cols,
             double **C_data, int *C_rows, int *C_cols);
void single_einsum(const double *data, int rows, int cols, const char *notation, const int *shape,
                   double **result_data, int *result_rows, int *result_cols,
                   int **new_shape, int *new_shape_size);
void reshape(const double *data, int data_rows, int data_cols, const int *shape, const int *new_shape,
             double **result_data, int *result_rows, int *result_cols);

#endif // COO_MATMUL_H
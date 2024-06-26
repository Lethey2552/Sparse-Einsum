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

// Custom hash function
struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = vec.size();
        for (auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Custom equality function
struct VectorEqual {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
        return a == b;
    }
};

void coo_bmm(const double *A_data, int A_rows, int A_cols,
             const double *B_data, int B_rows, int B_cols,
             double **C_data, int *C_rows, int *C_cols);
void single_einsum(const double *data, int rows, int cols, const char *notation,
                   double **result_data, int *result_rows, int *result_cols,
                   int **shape, int *shape_size);

void sum_over_trivial_indices(const double *data, int rows, int cols,
                              const std::string &input_notation, const std::string &output_notation,
                              std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map);
void calculate_diagonals(const std::string &input_notation, const std::vector<char> &diagonal_indices,
                         std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map);

#endif // COO_MATMUL_H
#ifndef COO_MATMUL_H
#define COO_MATMUL_H

#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>

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

void coo_matmul(double *A_data, int A_rows, int A_cols,
                double *B_data, int B_rows, int B_cols,
                double **C_data, int *C_rows, int *C_cols);

#endif // COO_MATMUL_H
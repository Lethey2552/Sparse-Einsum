#include "coo_methods.h"

void coo_matmul(double *A_data, int A_rows, int A_cols,
                double *B_data, int B_rows, int B_cols,
                double **C_data, int *C_rows, int *C_cols)
{

    std::unordered_map<std::pair<int, int>, double, PairHash, PairEqual> C_dict;

    // Iterate over A and B to perform matrix multiplication
    for (int i = 0; i < A_rows; ++i)
    {
        int row_A = static_cast<int>(A_data[i * A_cols]);
        int col_A = static_cast<int>(A_data[i * A_cols + 1]);
        double val_A = A_data[i * A_cols + 2];

        for (int j = 0; j < B_rows; ++j)
        {
            int col_B = static_cast<int>(B_data[j * B_cols]);
            int row_B = static_cast<int>(B_data[j * B_cols + 1]);
            double val_B = B_data[j * B_cols + 2];

            if (col_A == col_B)
            {
                auto key = std::make_pair(row_A, row_B);
                C_dict[key] += val_A * val_B;
            }
        }
    }

    // Convert C_dict to output format
    std::vector<std::tuple<int, int, double>> result_data;
    for (const auto &kvp : C_dict)
    {
        result_data.push_back(std::make_tuple(kvp.first.first, kvp.first.second, kvp.second));
    }

    // Default sort: first and second elements ascending
    std::sort(result_data.begin(), result_data.end());

    *C_rows = result_data.size();
    *C_cols = 3;

    *C_data = new double[*C_rows * *C_cols];
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        (*C_data)[i * *C_cols + 0] = std::get<0>(result_data[i]);
        (*C_data)[i * *C_cols + 1] = std::get<1>(result_data[i]);
        (*C_data)[i * *C_cols + 2] = std::get<2>(result_data[i]);
    }
}
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

// Function to perform COO block matrix multiplication
void coo_bmm(const double *A_data, int A_rows, int A_cols,
             const double *B_data, int B_rows, int B_cols,
             double **C_data, int *C_rows, int *C_cols)
{
    clock_t start, end;
    double cpu_time_used = 0;

    int cols_to_consider = A_cols - 3;

    // Extract unique values from the first column in A
    std::unordered_set<double> unique_values_set;
    for (int i = 0; i < A_rows; ++i)
    {
        unique_values_set.insert(A_data[i * A_cols]);
    }
    std::vector<double> unique_values(unique_values_set.begin(), unique_values_set.end());
    std::sort(unique_values.begin(), unique_values.end());

    // Preallocate space for masked arrays and AB_data
    int max_masked_size = std::max(A_rows, B_rows) * 3;
    double *A_masked = new double[max_masked_size];
    double *B_masked = new double[max_masked_size];

    std::vector<double> AB_data;
    AB_data.reserve(A_rows * B_rows * 3);

    int AB_data_rows = 0;
    int AB_data_cols = 0;

    // Iterate over unique values and perform matrix multiplication
    for (const double &unique_val : unique_values)
    {
        start = clock();

        int A_masked_size = 0;
        int B_masked_size = 0;

        // Compute masks
        for (int i = 0; i < A_rows; ++i)
        {
            if (A_data[i * A_cols] == unique_val)
            {
                std::copy(&A_data[i * A_cols + cols_to_consider], &A_data[(i + 1) * A_cols], &A_masked[A_masked_size]);
                A_masked_size += 3;
            }
        }

        for (int i = 0; i < B_rows; ++i)
        {
            if (B_data[i * B_cols] == unique_val)
            {
                std::copy(&B_data[i * B_cols + cols_to_consider], &B_data[(i + 1) * B_cols], &B_masked[B_masked_size]);
                B_masked_size += 3;
            }
        }

        end = clock();
        cpu_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;

        if (A_masked_size > 0 && B_masked_size > 0)
        {
            double *C_temp_data;
            int C_temp_rows, C_temp_cols;
            coo_matmul(A_masked, A_masked_size / 3, 3, B_masked, B_masked_size / 3, 3, &C_temp_data, &C_temp_rows, &C_temp_cols);

            // Reserve space for the current batch results
            AB_data.reserve(AB_data.size() + C_temp_rows * (C_temp_cols + 1)); // +1 for the batch dimension

            // Add batch dimensions to C_temp_data
            for (int i = 0; i < C_temp_rows; ++i)
            {
                AB_data.push_back(unique_val); // Add the unique value as the batch dimension
                AB_data.insert(AB_data.end(), &C_temp_data[i * C_temp_cols], &C_temp_data[(i + 1) * C_temp_cols]);
            }

            delete[] C_temp_data;

            AB_data_rows += C_temp_rows;
            AB_data_cols = C_temp_cols + 1; // 1 for the batch dimension
        }
    }

    delete[] A_masked;
    delete[] B_masked;

    std::cout << "Time: " << cpu_time_used << "s" << std::endl;

    *C_data = new double[AB_data.size()];
    std::copy(AB_data.begin(), AB_data.end(), *C_data);

    *C_rows = AB_data_rows;
    *C_cols = AB_data_cols;
}
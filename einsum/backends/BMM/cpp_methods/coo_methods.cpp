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

// Helper function to generate Cartesian product
template <typename T>
std::vector<std::vector<T>> cartesian_product(const std::vector<std::vector<T>> &v)
{
    std::vector<std::vector<T>> s = {{}};
    for (const auto &u : v)
    {
        std::vector<std::vector<T>> r;
        for (const auto &x : s)
        {
            for (const auto &y : u)
            {
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = std::move(r);
    }
    return s;
}

// Function to perform COO block matrix multiplication
void coo_bmm(const double *A_data, int A_rows, int A_cols,
             const double *B_data, int B_rows, int B_cols,
             double **C_data, int *C_rows, int *C_cols)
{
    clock_t start, end;
    double cpu_time_used;

    int cols_to_consider = A_cols - 3; // Exclude last 3 columns (data)

    std::vector<std::vector<double>> unique_values(cols_to_consider);

    // Extract unique values for each column in A
    for (int i = 0; i < cols_to_consider; ++i)
    {
        std::unordered_set<double> unique_vals_set;
        for (int j = 0; j < A_rows; ++j)
        {
            unique_vals_set.insert(A_data[j * A_cols + i]);
        }
        unique_values[i].assign(unique_vals_set.begin(), unique_vals_set.end());
    }

    // Generate combinations of unique values
    std::vector<std::vector<double>> combinations = cartesian_product(unique_values);
    std::sort(combinations.begin(), combinations.end());

    std::vector<double> AB_data;
    int AB_data_rows = 0;
    int AB_data_cols = 0;

    std::unordered_map<std::vector<double>, std::pair<std::vector<double>, std::vector<double>>, VectorHash, VectorEqual> masks_map;

    for (const auto &comb : combinations)
    {
        start = clock();
        std::vector<double> A_masked, B_masked;

        // Check if masks for current combination are already computed
        auto it = masks_map.find(comb);
        if (it == masks_map.end())
        {
            // Compute A_masked
            for (int i = 0; i < A_rows; ++i)
            {
                bool match = true;
                for (int j = 0; j < cols_to_consider; ++j)
                {
                    if (A_data[i * A_cols + j] != comb[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    A_masked.insert(A_masked.end(), &A_data[i * A_cols + cols_to_consider], &A_data[(i + 1) * A_cols]);
                }
            }

            // Compute B_masked
            for (int i = 0; i < B_rows; ++i)
            {
                bool match = true;
                for (int j = 0; j < cols_to_consider; ++j)
                {
                    if (B_data[i * B_cols + j] != comb[j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    B_masked.insert(B_masked.end(), &B_data[i * B_cols + cols_to_consider], &B_data[(i + 1) * B_cols]);
                }
            }

            // Store computed masks in the map
            masks_map[comb] = std::make_pair(A_masked, B_masked);
        }
        else
        {
            // Masks already computed, retrieve them
            A_masked = it->second.first;
            B_masked = it->second.second;
        }
        end = clock();
        cpu_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;

        if (!A_masked.empty() && !B_masked.empty())
        {
            double *C_temp_data;
            int C_temp_rows, C_temp_cols;
            coo_matmul(A_masked.data(), A_masked.size() / 3, 3, B_masked.data(), B_masked.size() / 3, 3, &C_temp_data, &C_temp_rows, &C_temp_cols);

            // Add batch dimensions to C_temp_data
            std::vector<double> comb_with_values;
            for (int i = 0; i < C_temp_rows; ++i)
            {
                comb_with_values.insert(comb_with_values.end(), comb.begin(), comb.end());
                comb_with_values.insert(comb_with_values.end(), &C_temp_data[i * C_temp_cols], &C_temp_data[(i + 1) * C_temp_cols]);
            }

            AB_data.insert(AB_data.end(), comb_with_values.begin(), comb_with_values.end());
            delete[] C_temp_data;

            AB_data_rows += C_temp_rows;
            AB_data_cols = C_temp_cols + cols_to_consider;
        }
    }
    std::cout << "Time: " << cpu_time_used << "s" << std::endl;

    *C_data = new double[AB_data.size()];
    std::copy(AB_data.begin(), AB_data.end(), *C_data);

    *C_rows = AB_data_rows;
    *C_cols = AB_data_cols;
}
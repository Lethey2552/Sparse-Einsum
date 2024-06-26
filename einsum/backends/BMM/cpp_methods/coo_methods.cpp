#include "coo_methods.h"

double cpu_time_used = 0;

void coo_bmm(const double *A_data, int A_rows, int A_cols,
            const double *B_data, int B_rows, int B_cols,
            double **C_data, int *C_rows, int *C_cols)
{
    // clock_t start, end;
    // double cpu_time_used = 0;
    // start = clock();
    // end = clock();
    // cpu_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "BMM Time: " << cpu_time_used << "s" << std::endl;

    // Check if A_data and B_data are batched
    // If batched extract batches else, treat as single matrices
    bool is_batched = (A_cols > 3 && B_cols > 3);

    std::unordered_map<double, std::vector<std::tuple<int, int, double>>> A_batches;
    std::unordered_map<double, std::vector<std::tuple<int, int, double>>> B_batches;

    if (is_batched) {
        // Separate A and B entries by batch dimension
        for (int i = 0; i < A_rows; ++i)
        {
            double batch = A_data[i * A_cols];
            int row_A = static_cast<int>(A_data[i * A_cols + 1]);
            int col_A = static_cast<int>(A_data[i * A_cols + 2]);
            double val_A = A_data[i * A_cols + 3];
            A_batches[batch].emplace_back(row_A, col_A, val_A);
        }

        for (int i = 0; i < B_rows; ++i)
        {
            double batch = B_data[i * B_cols];
            int col_B = static_cast<int>(B_data[i * B_cols + 1]);
            int row_B = static_cast<int>(B_data[i * B_cols + 2]);
            double val_B = B_data[i * B_cols + 3];
            B_batches[batch].emplace_back(col_B, row_B, val_B);
        }
    } else {
        // Treat A and B as single matrices
        A_batches[0.0].reserve(A_rows);
        B_batches[0.0].reserve(B_rows);

        for (int i = 0; i < A_rows; ++i)
        {
            A_batches[0.0].emplace_back(static_cast<int>(A_data[i * A_cols + 1]),
                                        static_cast<int>(A_data[i * A_cols + 2]),
                                        A_data[i * A_cols + 3]);
        }

        for (int i = 0; i < B_rows; ++i)
        {
            B_batches[0.0].emplace_back(static_cast<int>(B_data[i * B_cols + 1]),
                                        static_cast<int>(B_data[i * B_cols + 2]),
                                        B_data[i * B_cols + 3]);
        }
    }

    std::vector<std::tuple<double, int, int, double>> result_data;

    // Perform matrix multiplication for each batch
    for (const auto &batch_pair : A_batches)
    {
        double batch = batch_pair.first;
        const auto &A_batch = batch_pair.second;
        const auto &B_batch = B_batches[batch];

        std::unordered_map<std::pair<int, int>, double, PairHash, PairEqual> C_dict;

        for (const auto &a : A_batch)
        {
            int row_A = std::get<0>(a);
            int col_A = std::get<1>(a);
            double val_A = std::get<2>(a);

            for (const auto &b : B_batch)
            {
                int col_B = std::get<0>(b);
                int row_B = std::get<1>(b);
                double val_B = std::get<2>(b);

                if (col_A == col_B)
                {
                    auto key = std::make_pair(row_A, row_B);
                    C_dict[key] += val_A * val_B;
                }
            }
        }

        for (const auto &kvp : C_dict)
        {
            result_data.push_back(std::make_tuple(batch, kvp.first.first, kvp.first.second, kvp.second));
        }
    }

    std::sort(result_data.begin(), result_data.end());

    *C_rows = result_data.size();
    *C_cols = is_batched ? 4 : 3; // 4 columns if batched (batch, row, col, value); otherwise 3 (row, col, value)

    *C_data = new double[*C_rows * *C_cols];
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        if (is_batched) {
            (*C_data)[i * *C_cols + 0] = std::get<0>(result_data[i]);
            (*C_data)[i * *C_cols + 1] = std::get<1>(result_data[i]);
            (*C_data)[i * *C_cols + 2] = std::get<2>(result_data[i]);
            (*C_data)[i * *C_cols + 3] = std::get<3>(result_data[i]);
        } else {
            (*C_data)[i * *C_cols + 0] = std::get<1>(result_data[i]);
            (*C_data)[i * *C_cols + 1] = std::get<2>(result_data[i]);
            (*C_data)[i * *C_cols + 2] = std::get<3>(result_data[i]);
        }
    }
}

// Helper function to sum over trivially removable indices
void sum_over_trivial_indices(const double *data, int rows, int cols,
                              const std::string &input_notation, const std::string &output_notation,
                              std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map) {
    // Construct indices map for easy lookup
    std::unordered_map<char, int> indices;
    indices.reserve(input_notation.size());
    for (int i = 0; i < input_notation.size(); ++i) {
        indices[input_notation[i]] = i;
    }

    // Process each row according to output notation
    for (int i = 0; i < rows; ++i) {
        std::vector<int> key;

        // Construct key based on output indices
        for (char idx : output_notation) {
            int index = indices[idx];
            key.push_back(static_cast<int>(data[i * cols + index]));
        }

        // Find the value column index in data
        int value_index = cols - 1;
        double value = data[i * cols + value_index];

        // Insert into result_map, summing over trivially removable indices
        result_map[key] += value;
    }
}

void single_einsum(const double *data, int rows, int cols, const char *notation,
                   double **result_data, int *result_rows, int *result_cols,
                   int **shape, int *shape_size) {
    std::string notation_str(notation);
    auto arrow_pos = notation_str.find("->");
    std::string input_notation = notation_str.substr(0, arrow_pos);
    std::string output_notation = notation_str.substr(arrow_pos + 2);

    assert(input_notation.find(',') == std::string::npos);

    // Determine if the input notation indicates diagonal computation
    bool is_diagonal = false;
    std::unordered_map<char, int> input_indices_count;
    
    // Count occurrences of each index in input notation
    for (char idx : input_notation) {
        if (input_indices_count.find(idx) == input_indices_count.end()) {
            input_indices_count[idx] = 1;
        } else {
            input_indices_count[idx]++;
        }
    }

    std::vector<char> diagonal_indices;
    // Check for repeated indices indicating diagonal computation
    for (const auto& entry : input_indices_count) {
        if (entry.second > 1) {
            is_diagonal = true;
            diagonal_indices.push_back(entry.first);
        }
    }

    std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> result_map;
    
    // Sum over trivially removable indices
    sum_over_trivial_indices(data, rows, cols, input_notation, output_notation, result_map);

    //TODO: Check the sum_over_trivial_indices result and implement the diagonal calculation 
    // as well as the swapping of indices in that exact order.

    // Convert for sorting
    std::vector<std::tuple<std::vector<int>, double>> sorted_results;
    sorted_results.reserve(result_map.size());
    for (auto &entry : result_map) {
        sorted_results.emplace_back(std::move(entry.first), entry.second);
    }

    std::sort(sorted_results.begin(), sorted_results.end());

    // Prepare the output data
    *result_rows = sorted_results.size();
    *result_cols = output_notation.size() + 1;  // +1 for the value column
    *result_data = new double[*result_rows * *result_cols];

    // Initialize shape array
    *shape_size = output_notation.size();
    *shape = new int[*shape_size]();
    
    int r = 0;
    for (const auto &entry : sorted_results) {
        for (size_t c = 0; c < std::get<0>(entry).size(); ++c) {
            int value = std::get<0>(entry)[c] + 1;
            (*result_data)[r * *result_cols + c] = value - 1;
            if (value > (*shape)[c]) {
                (*shape)[c] = value;
            }
        }
        (*result_data)[r * *result_cols + output_notation.size()] = std::get<1>(entry);
        ++r;
    }
}
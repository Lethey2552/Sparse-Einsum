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

    if (is_batched)
    {
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
    }
    else
    {
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
        if (is_batched)
        {
            (*C_data)[i * *C_cols + 0] = std::get<0>(result_data[i]);
            (*C_data)[i * *C_cols + 1] = std::get<1>(result_data[i]);
            (*C_data)[i * *C_cols + 2] = std::get<2>(result_data[i]);
            (*C_data)[i * *C_cols + 3] = std::get<3>(result_data[i]);
        }
        else
        {
            (*C_data)[i * *C_cols + 0] = std::get<1>(result_data[i]);
            (*C_data)[i * *C_cols + 1] = std::get<2>(result_data[i]);
            (*C_data)[i * *C_cols + 2] = std::get<3>(result_data[i]);
        }
    }
}

// Helper function to find the positions of each character in a string
std::unordered_map<char, std::vector<int>> find_positions(const std::string &str)
{
    std::unordered_map<char, std::vector<int>> positions;
    for (int i = 0; i < str.size(); ++i)
    {
        positions[str[i]].push_back(i);
    }
    return positions;
}

// Function to find sum indices and update input_notation
void find_sum_indices(std::string &input_notation, const std::string &output_notation,
                      std::vector<int> &sum_indices)
{
    std::unordered_map<char, int> input_count;
    for (char idx : input_notation)
    {
        input_count[idx]++;
    }

    sum_indices.clear(); // Clear existing sum indices

    // Identify sum indices and update input_notation
    for (int i = input_notation.size() - 1; i >= 0; --i)
    {
        char idx = input_notation[i];

        if (input_count[idx] == 1 && output_notation.find(idx) == std::string::npos)
        {
            sum_indices.push_back(i);
            input_notation.erase(i, 1); // Remove character at index i from input_notation
        }
    }
}

// Function to find diagonal indices and update input_notation
void find_diag_indices(std::string &input_notation, std::vector<int> &diag_indices)
{
    std::unordered_map<char, std::vector<int>> positions;
    for (int i = 0; i < input_notation.size(); ++i)
    {
        positions[input_notation[i]].push_back(i);
    }

    diag_indices.clear(); // Clear diag_indices to avoid duplicate entries

    for (const auto &entry : positions)
    {
        if (entry.second.size() > 1)
        {
            diag_indices.insert(diag_indices.end(), entry.second.begin(), entry.second.end());

            // Remove indices from input_notation
            for (size_t i = entry.second.size() - 1; i > 0; --i)
            {
                input_notation.erase(entry.second[i], 1);
            }
        }
    }
}

// Function to find permutation indices
void find_perm_indices(const std::string &input_notation, const std::string &output_notation,
                       std::vector<int> &perm_indices)
{
    // Map to store indices of output_notation
    std::unordered_map<char, int> output_indices;
    for (int i = 0; i < output_notation.size(); ++i)
    {
        output_indices[output_notation[i]] = i;
    }

    perm_indices.clear();
    for (int i = 0; i < input_notation.size(); ++i)
    {
        char idx = input_notation[i];
        if (output_indices.find(idx) != output_indices.end())
        {
            perm_indices.push_back(output_indices[idx]);
        }
    }
}

void sum_over_trivial_indices(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
                              const std::vector<int> &sum_indices)
{

    std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> new_result_map;

    for (const auto &entry : result_map)
    {
        const std::vector<int> &key = entry.first;
        double value = entry.second;
        std::vector<int> new_key;

        for (size_t i = 0; i < key.size(); ++i)
        {
            if (std::find(sum_indices.begin(), sum_indices.end(), i) == sum_indices.end())
            {
                new_key.push_back(key[i]);
            }
        }

        new_result_map[new_key] += value;
    }

    // Assign new_result_map to result_map
    result_map = std::move(new_result_map);
}

void remove_non_diag_indices(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
                             const std::vector<int> &diag_indices)
{
    std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> new_result_map;

    // Iterate over each entry in result_map
    for (const auto &entry : result_map)
    {
        const std::vector<int> &key = entry.first;
        bool is_valid_diagonal = true;

        // Check if all indices in diag_indices have the same value in the key
        for (size_t i = 1; i < diag_indices.size(); ++i)
        {
            if (key[diag_indices[i]] != key[diag_indices[0]])
            {
                is_valid_diagonal = false;
                break;
            }
        }

        // If valid diagonal entry, construct new_key with only the last diag_index removed
        if (is_valid_diagonal)
        {
            std::vector<int> new_key;
            bool removed_last_diag = false;
            for (int i = key.size() - 1; i >= 0; --i)
            {
                if (!removed_last_diag && std::find(diag_indices.begin(), diag_indices.end(), i) != diag_indices.end())
                {
                    removed_last_diag = true;
                }
                else
                {
                    new_key.push_back(key[i]);
                }
            }
            std::reverse(new_key.begin(), new_key.end()); // Reverse new_key to correct order
            new_result_map[new_key] = entry.second;
        }
    }

    // Assign new_result_map to result_map
    result_map = std::move(new_result_map);
}

void apply_permutation(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
                       const std::vector<int> &perm_indices)
{
    std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> new_result_map;

    for (const auto &entry : result_map)
    {
        const std::vector<int> &key = entry.first;
        std::vector<int> new_key(key.size());

        // Apply permutation to create new_key
        for (size_t i = 0; i < perm_indices.size(); ++i)
        {
            new_key[perm_indices[i]] = key[i];
        }

        new_result_map[new_key] = entry.second;
    }

    result_map = std::move(new_result_map);
}

void result_map_to_data(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
                        std::string &output_notation,
                        double **result_data, int *result_rows, int *result_cols,
                        int **new_shape, int *new_shape_size)
{
    // Convert for sorting
    std::vector<std::tuple<std::vector<int>, double>> sorted_results;
    sorted_results.reserve(result_map.size());
    for (auto &entry : result_map)
    {
        sorted_results.emplace_back(std::move(entry.first), entry.second);
    }

    std::sort(sorted_results.begin(), sorted_results.end());

    // Prepare the output data
    *result_rows = sorted_results.size();
    *result_cols = output_notation.size() + 1; // +1 for the value column
    *result_data = new double[*result_rows * *result_cols];

    // Initialize shape array
    *new_shape_size = output_notation.size();
    *new_shape = new int[*new_shape_size]();

    int r = 0;
    for (const auto &entry : sorted_results)
    {
        for (size_t c = 0; c < std::get<0>(entry).size(); ++c)
        {
            int value = std::get<0>(entry)[c] + 1;
            (*result_data)[r * *result_cols + c] = value - 1;
            if (value > (*new_shape)[c])
            {
                (*new_shape)[c] = value;
            }
        }
        (*result_data)[r * *result_cols + output_notation.size()] = std::get<1>(entry);
        ++r;
    }
}

void single_einsum(const double *data, int rows, int cols,
                   const char *notation, const int *shape,
                   double **result_data, int *result_rows, int *result_cols,
                   int **new_shape, int *new_shape_size)
{

    bool debug = false;

    std::string notation_str(notation);
    auto arrow_pos = notation_str.find("->");
    std::string input_notation = notation_str.substr(0, arrow_pos);
    std::string output_notation = notation_str.substr(arrow_pos + 2);

    assert(input_notation.find(',') == std::string::npos);

    std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> result_map;

    // Handle equal input and output
    if (input_notation == output_notation)
    {
        *result_rows = rows;
        *result_cols = cols;
        *new_shape_size = output_notation.size();

        *new_shape = new int[*new_shape_size];
        for (int i = 0; i < *new_shape_size; ++i)
        {
            (*new_shape)[i] = shape[i];
        }

        *result_data = new double[rows * cols];
        std::copy(data, data + rows * cols, *result_data);
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        std::vector<int> key;
        for (int j = 0; j < cols - 1; ++j)
        {
            key.push_back(static_cast<int>(data[i * cols + j]));
        }
        double value = data[i * cols + cols - 1];
        result_map[key] += value;
    }

    std::vector<int> sum_indices;
    std::vector<int> diag_indices;
    std::vector<int> perm_indices;

    // Handle diagonal indices
    find_diag_indices(input_notation, diag_indices);

    if (debug)
    {
        std::cout << "diag_indices: " << std::endl;
        for (auto i : diag_indices)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "New notation: " << input_notation << "\n"
                  << std::endl;
    }

    remove_non_diag_indices(result_map, diag_indices);

    if (debug)
    {
        // Print the altered result_map
        std::cout << "Altered result_map:" << std::endl;
        for (const auto &entry : result_map)
        {
            std::cout << "{";
            for (size_t i = 0; i < entry.first.size(); ++i)
            {
                std::cout << entry.first[i];
                if (i != entry.first.size() - 1)
                {
                    std::cout << ", ";
                }
            }
            std::cout << "} => " << entry.second << std::endl;
        }
    }

    find_sum_indices(input_notation, output_notation, sum_indices);

    if (debug)
    {
        std::cout << "Sum indices: " << std::endl;
        for (auto i : sum_indices)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "New notation: " << input_notation << "\n"
                  << std::endl;
    }

    sum_over_trivial_indices(result_map, sum_indices);

    if (debug)
    {
        // Print the altered result_map
        std::cout << "Altered result_map:" << std::endl;
        for (const auto &entry : result_map)
        {
            std::cout << "{";
            for (size_t i = 0; i < entry.first.size(); ++i)
            {
                std::cout << entry.first[i];
                if (i != entry.first.size() - 1)
                {
                    std::cout << ", ";
                }
            }
            std::cout << "} => " << entry.second << std::endl;
        }
    }

    find_perm_indices(input_notation, output_notation, perm_indices);

    if (debug)
    {
        std::cout << "perm_indices: " << std::endl;
        for (auto i : perm_indices)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }

    apply_permutation(result_map, perm_indices);

    if (debug)
    {
        // Print the altered result_map
        std::cout << "Altered result_map:" << std::endl;
        for (const auto &entry : result_map)
        {
            std::cout << "{";
            for (size_t i = 0; i < entry.first.size(); ++i)
            {
                std::cout << entry.first[i];
                if (i != entry.first.size() - 1)
                {
                    std::cout << ", ";
                }
            }
            std::cout << "} => " << entry.second << std::endl;
        }
    }

    // Convert for sorting
    std::vector<std::tuple<std::vector<int>, double>> sorted_results;
    sorted_results.reserve(result_map.size());
    for (auto &entry : result_map)
    {
        sorted_results.emplace_back(std::move(entry.first), entry.second);
    }

    std::sort(sorted_results.begin(), sorted_results.end());

    // Prepare the output data
    *result_rows = sorted_results.size();
    *result_cols = output_notation.size() + 1; // +1 for the value column
    *result_data = new double[*result_rows * *result_cols];

    // Initialize shape array
    *new_shape_size = output_notation.size();
    *new_shape = new int[*new_shape_size]();

    int r = 0;
    for (const auto &entry : sorted_results)
    {
        for (size_t c = 0; c < std::get<0>(entry).size(); ++c)
        {
            int value = std::get<0>(entry)[c] + 1;
            (*result_data)[r * *result_cols + c] = value - 1;
            if (value > (*new_shape)[c])
            {
                (*new_shape)[c] = value;
            }
        }
        (*result_data)[r * *result_cols + output_notation.size()] = std::get<1>(entry);
        ++r;
    }
}
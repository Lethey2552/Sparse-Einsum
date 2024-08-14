#include "coo_methods.h"

double cpu_time_used = 0;
double diagonal_time = 0;
double sum_time = 0;
double permute_time = 0;
double sort_time = 0;
double bmm_sort_time = 0;

// void coo_bmm_test(const double *A_data, int A_rows, int A_cols,
//                   const double *B_data, int B_rows, int B_cols,
//                   double **C_data, int *C_rows, int *C_cols)
// {

//     bool is_batched = (A_cols > 3 && B_cols > 3);
//     is_batched = true;
//     // std::cout << is_batched << std::endl;

//     // Prepare the result container
//     std::vector<std::tuple<double, int, int, double>> result_data;

//     int max_batch = 1;

//     for (int i = 0; i < A_rows; ++i)
//     {
//         int batch_idx = static_cast<int>(A_data[i * A_cols]);
//         max_batch = (max_batch < batch_idx ? batch_idx : max_batch);
//     }

//     // Iterate over each batch
//     for (int b = 0; b < (is_batched ? max_batch : 1); ++b)
//     {
//         int A_batch_start = b * A_cols;
//         int B_batch_start = b * B_cols;

//         // Extract A and B's coordinates and values
//         std::vector<int> A1_crd, A2_crd;
//         std::vector<double> A_vals;
//         std::vector<int> B1_crd, B2_crd;
//         std::vector<double> B_vals;

//         if (b == 0)
//             std::cout << "PRINT A:" << std::endl;
//         for (int i = 0; i < A_rows; ++i)
//         {
//             int A_batch = static_cast<int>(A_data[i * A_cols]);

//             if (A_batch != b)
//                 continue;

//             int A1 = static_cast<int>(A_data[i * A_cols + 1]);
//             int A2 = static_cast<int>(A_data[i * A_cols + 2]);
//             double A_val = A_data[i * A_cols + 3];

//             A1_crd.push_back(A1);
//             A2_crd.push_back(A2);
//             A_vals.push_back(A_val);

//             // Print values
//             if (b == 0)
//                 std::cout << "[" << A1 << " " << A2 << " " << A_val << "]" << std::endl;
//         }

//         if (b == 0)
//             std::cout << "PRINT B:" << std::endl;
//         for (int j = 0; j < B_rows; ++j)
//         {
//             int B_batch = static_cast<int>(B_data[j * B_cols]);

//             if (B_batch != b)
//                 continue;

//             int B1 = static_cast<int>(B_data[j * B_cols + 1]);
//             int B2 = static_cast<int>(B_data[j * B_cols + 2]);
//             double B_val = B_data[j * B_cols + 3];

//             B1_crd.push_back(B1);
//             B2_crd.push_back(B2);
//             B_vals.push_back(B_val);

//             // Print values
//             if (b == 0)
//                 std::cout << "[" << B1 << " " << B2 << " " << B_val << "]" << std::endl;
//         }

//         if (b == 0)
//         { // Perform the sparse matrix multiplication
//             int iA = 0, jB = 0;
//             while (iA < A_rows && jB < B_rows)
//             {
//                 int row_A = A1_crd[iA];
//                 int col_B = B1_crd[jB];

//                 std::cout << "ROW A: " << row_A << std::endl;

//                 if (row_A == col_B)
//                 {
//                     // Iterate over matching indices
//                     int seg_A = iA;
//                     while (seg_A < A_rows && A1_crd[seg_A] == row_A)
//                         seg_A++;
//                     int seg_B = jB;
//                     while (seg_B < B_rows && B1_crd[seg_B] == col_B)
//                         seg_B++;

//                     // Calculate the value for C
//                     for (int i = iA; i < seg_A; ++i)
//                     {
//                         int col_A = A2_crd[i];
//                         for (int j = jB; j < seg_B; ++j)
//                         {
//                             if (B2_crd[j] == col_A)
//                             {
//                                 double value = A_vals[i] * B_vals[j];
//                                 result_data.push_back(std::make_tuple(static_cast<double>(b), row_A, B2_crd[j], value));
//                             }
//                         }
//                     }
//                     iA = seg_A;
//                     jB = seg_B;
//                 }
//                 else if (row_A < col_B)
//                 {
//                     iA++;
//                 }
//                 else
//                 {
//                     jB++;
//                 }
//             }
//         }
//     }

//     // Sort the result by batch, row, and column indices
//     std::sort(result_data.begin(), result_data.end());

//     *C_rows = static_cast<int>(result_data.size());
//     *C_cols = is_batched ? 4 : 3;

//     *C_data = new double[*C_rows * *C_cols];
//     for (size_t i = 0; i < result_data.size(); ++i)
//     {
//         if (is_batched)
//         {
//             (*C_data)[i * *C_cols + 0] = std::get<0>(result_data[i]);
//             (*C_data)[i * *C_cols + 1] = std::get<1>(result_data[i]);
//             (*C_data)[i * *C_cols + 2] = std::get<2>(result_data[i]);
//             (*C_data)[i * *C_cols + 3] = std::get<3>(result_data[i]);
//         }
//         else
//         {
//             (*C_data)[i * *C_cols + 0] = std::get<1>(result_data[i]);
//             (*C_data)[i * *C_cols + 1] = std::get<2>(result_data[i]);
//             (*C_data)[i * *C_cols + 2] = std::get<3>(result_data[i]);
//         }
//     }

//     // Print the result_data
//     std::cout << "TEST C_data:" << std::endl;
//     for (size_t i = 0; i < *C_rows; ++i)
//     {
//         std::cout << "(";
//         if (is_batched)
//         {
//             std::cout << (*C_data)[i * *C_cols + 0] << ", " // Batch index
//                       << (*C_data)[i * *C_cols + 1] << ", " // Row index
//                       << (*C_data)[i * *C_cols + 2] << ", " // Column index
//                       << (*C_data)[i * *C_cols + 3];        // Value
//         }
//         else
//         {
//             std::cout << (*C_data)[i * *C_cols + 0] << ", " // Row index
//                       << (*C_data)[i * *C_cols + 1] << ", " // Column index
//                       << (*C_data)[i * *C_cols + 2];        // Value
//         }
//         std::cout << ")" << std::endl;
//     }
// }

void coo_bmm(const double *A_data, int A_rows, int A_cols,
             const double *B_data, int B_rows, int B_cols,
             double **C_data, int *C_rows, int *C_cols)
{
    clock_t start, end;
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

    // start = clock();
    std::sort(std::execution::par, result_data.begin(), result_data.end());
    // std::sort(result_data.begin(), result_data.end());
    // end = clock();
    // bmm_sort_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "bmm_sort_time: " << bmm_sort_time << "s" << std::endl;

    *C_rows = static_cast<int>(result_data.size());
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

// Function to find sum indices and update input_chars
void find_sum_indices(std::vector<std::string> &input_chars, const std::vector<std::string> &output_chars,
                      std::vector<int> &sum_indices, std::vector<int> &shape)
{
    std::unordered_map<std::string, int> input_count;
    for (const auto &idx : input_chars)
    {
        input_count[idx]++;
    }

    std::unordered_set<std::string> output_set(output_chars.begin(), output_chars.end());

    sum_indices.clear();
    sum_indices.reserve(input_chars.size());

    std::vector<std::string> new_input_chars;
    new_input_chars.reserve(input_chars.size());
    std::vector<int> new_shape;
    new_shape.reserve(shape.size());

    for (int i = 0; i < input_chars.size(); ++i)
    {
        const auto &idx = input_chars[i];
        if (input_count[idx] == 1 && output_set.find(idx) == output_set.end())
        {
            sum_indices.push_back(i);
        }
        else
        {
            new_input_chars.push_back(idx);
            new_shape.push_back(shape[i]);
        }
    }

    input_chars = std::move(new_input_chars);
    shape = std::move(new_shape);
}

// Function to find diagonal indices in UTF-8 characters notation
void find_diag_indices(std::vector<std::string> &input_chars, std::vector<int> &diag_indices, std::vector<int> &shape)
{
    std::unordered_map<std::string, std::vector<int>> positions;
    for (int i = 0; i < input_chars.size(); ++i)
    {
        positions[input_chars[i]].push_back(i);
    }

    diag_indices.clear();

    for (const auto &entry : positions)
    {
        if (entry.second.size() > 1)
        {
            diag_indices.insert(diag_indices.end(), entry.second.begin(), entry.second.end());

            // Remove indices from input_chars and update shape
            for (size_t i = entry.second.size() - 1; i > 0; --i)
            {
                input_chars.erase(input_chars.begin() + entry.second[i]);
                shape.erase(shape.begin() + entry.second[i]);
            }
        }
    }
}

// Function to find permutation indices with UTF-8 characters
void find_perm_indices(const std::vector<std::string> &input_chars, const std::vector<std::string> &output_chars,
                       std::vector<int> &perm_indices, std::vector<int> &shape)
{
    if (input_chars == output_chars)
    {
        return;
    }

    // Map to store indices of output_chars
    std::unordered_map<std::string, int> output_indices;
    for (int i = 0; i < output_chars.size(); ++i)
    {
        output_indices[output_chars[i]] = i;
    }

    perm_indices.clear();
    std::vector<int> new_shape(output_chars.size());

    for (int i = 0; i < input_chars.size(); ++i)
    {
        const std::string &idx = input_chars[i];
        if (output_indices.find(idx) != output_indices.end())
        {
            int permuted_index = output_indices[idx];
            perm_indices.push_back(permuted_index);
            new_shape[permuted_index] = shape[i];
        }
    }

    shape = std::move(new_shape); // Assign the permuted shape back
}

// Function to iterate through UTF-8 encoded string
std::vector<std::string> split_utf8(const std::string &utf8_str)
{
    std::vector<std::string> utf8_chars;
    size_t i = 0;

    while (i < utf8_str.size())
    {
        unsigned char c = utf8_str[i];
        size_t char_len = 1;

        if ((c & 0x80) == 0x00)
        {
            char_len = 1;
        }
        else if ((c & 0xE0) == 0xC0)
        {
            char_len = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            char_len = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            char_len = 4;
        }

        utf8_chars.push_back(utf8_str.substr(i, char_len));
        i += char_len;
    }

    return utf8_chars;
}

void fill_dimensions_and_entries(const double *data, int rows, int cols,
                                 std::vector<int> &dimensions, std::vector<Entry> &entries)
{
    dimensions.reserve(rows * (cols - 1));

    for (int i = 0; i < rows; ++i)
    {
        Entry entry;
        entry.offset = static_cast<uint32_t>(dimensions.size()); // Offset before adding new elements
        entry.num_idx = cols - 1;

        // Fill dimensions vector
        for (int j = 0; j < cols - 1; ++j)
        {
            dimensions.push_back(static_cast<int>(data[i * cols + j]));
        }

        entry.value = data[i * cols + cols - 1];
        entries.push_back(entry);
    }
}

void remove_non_diag_indices(std::vector<int> &dimensions, std::vector<Entry> &entries,
                             const std::vector<int> &diag_indices, int cols)
{
    std::unordered_set<int> diag_indices_set(diag_indices.begin(), diag_indices.end());
    std::vector<Entry> new_entries;
    std::vector<int> new_dimensions;
    new_dimensions.reserve(dimensions.size()); // Reserve space based on the original dimensions

    // Iterate over existing entries
    for (const auto &entry : entries)
    {
        int offset = entry.offset;
        int num_idx = entry.num_idx;
        bool is_valid_diagonal = true;

        // Check if all indices in diag_indices have the same value in the dimensions vector
        for (size_t i = 1; i < diag_indices.size(); ++i)
        {
            if (dimensions[offset + diag_indices[i]] != dimensions[offset + diag_indices[0]])
            {
                is_valid_diagonal = false;
                break;
            }
        }

        if (is_valid_diagonal)
        {
            Entry new_entry;
            new_entry.offset = static_cast<uint32_t>(new_dimensions.size());
            new_entry.num_idx = num_idx - 1;
            new_entry.value = entry.value;

            // Copy the dimensions except the last diag_index
            for (int i = 0; i < num_idx; ++i)
            {
                if (i != diag_indices.back())
                {
                    new_dimensions.push_back(dimensions[offset + i]);
                }
            }

            new_entries.push_back(new_entry);
        }
    }

    // Update the original dimensions and entries with new data
    dimensions = std::move(new_dimensions);
    entries = std::move(new_entries);
}

void apply_permutation(std::vector<int> &dimensions, std::vector<Entry> &entries, const std::vector<int> &perm_indices)
{
    // Parallelize the loop over entries using OpenMP
    // #pragma omp parallel for
    for (int i = 0; i < entries.size(); ++i)
    {
        int offset = entries[i].offset;
        int num_idx = entries[i].num_idx;

        std::vector<int> new_dimensions(num_idx);

        // Apply permutation to create new_dimensions
        for (size_t j = 0; j < perm_indices.size(); ++j)
        {
            new_dimensions[perm_indices[j]] = dimensions[offset + j];
        }

        // Update dimensions with new_dimensions
        for (size_t j = 0; j < perm_indices.size(); ++j)
        {
            dimensions[offset + j] = new_dimensions[j];
        }
    }
}

void sum_over_trivial_indices(std::vector<int> &dimensions,
                              std::vector<Entry> &entries,
                              const std::vector<int> &sum_indices,
                              int cols)
{
    // Create a bitset for sum indices for fast lookup
    std::vector<bool> is_sum_index(cols, false);
    for (int index : sum_indices)
    {
        is_sum_index[index] = true;
    }

    std::unordered_map<std::vector<int>, double, ArrayHash> key_to_value;

// clock_t start, end;
// double total_time = 0;

// Parallel region
#pragma omp parallel
    {
        // Each thread needs its own local data to avoid contention
        std::unordered_map<std::vector<int>, double, ArrayHash> local_key_to_value;
        std::vector<int> local_temp_key;
        local_temp_key.reserve(cols);

#pragma omp for schedule(static, 1000)
        for (int i = 0; i < entries.size(); ++i)
        {
            const auto &entry = entries[i];
            int offset = entry.offset;
            int num_idx = entry.num_idx;
            double value = entry.value;

            // Clear and construct new key excluding sum indices
            local_temp_key.clear();
            for (int j = 0; j < num_idx; ++j)
            {
                if (!is_sum_index[j])
                {
                    local_temp_key.push_back(dimensions[offset + j]);
                }
            }

            // Aggregate values for each unique key
            local_key_to_value[local_temp_key] += value;
        }

// Merge local results into the global map
#pragma omp critical
        {
            // start = clock();
            for (const auto &[key, value] : local_key_to_value)
            {
                key_to_value[key] += value;
            }
            // end = clock();
            // total_time += ((double)(end - start)) / CLOCKS_PER_SEC;
        }
    }

    // std::cout << "TOTAL THREAD TIME: " << total_time << std::endl;
    // std::cout << entries.size() << std::endl;

    // Convert the hashmap back into the entry and dimensions format
    std::vector<Entry> new_entries;
    std::vector<int> new_dimensions;

    new_entries.reserve(key_to_value.size());
    new_dimensions.reserve(dimensions.size());

    for (const auto &[key, value] : key_to_value)
    {
        Entry new_entry;
        new_entry.offset = static_cast<uint32_t>(new_dimensions.size());
        new_entry.num_idx = static_cast<uint16_t>(key.size());
        new_entry.value = value;

        new_dimensions.insert(new_dimensions.end(), key.begin(), key.end());
        new_entries.push_back(new_entry);
    }

    // Update original entries and dimensions with new data
    entries = std::move(new_entries);
    dimensions = std::move(new_dimensions);
}

// Helper function to compare entries for sorting
bool compare_entries(const Entry &a, const Entry &b, const std::vector<int> &dimensions)
{
    return std::lexicographical_compare(
        dimensions.begin() + a.offset, dimensions.begin() + a.offset + a.num_idx,
        dimensions.begin() + b.offset, dimensions.begin() + b.offset + b.num_idx);
}

void parallel_sort(std::vector<Entry> &entries, const std::vector<int> &dimensions)
{
    int num_threads = omp_get_max_threads();
    size_t n = entries.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

// Sort chunks in parallel
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        if (start < end)
        {
            std::sort(entries.begin() + start, entries.begin() + end,
                      [&dimensions](const Entry &a, const Entry &b)
                      {
                          return compare_entries(a, b, dimensions);
                      });
        }
    }

    // Merge sorted chunks
    for (size_t step = chunk_size; step < n; step *= 2)
    {
#pragma omp parallel for
        for (int64_t start = 0; static_cast<size_t>(start) < n; start += 2 * step)
        {
            int64_t mid = std::min(start + step, n);
            int64_t end = std::min(start + 2 * step, n);
            std::inplace_merge(entries.begin() + start, entries.begin() + mid, entries.begin() + end,
                               [&dimensions](const Entry &a, const Entry &b)
                               {
                                   return compare_entries(a, b, dimensions);
                               });
        }
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

    // Handle equal input and output
    if (input_notation == output_notation)
    {
        *result_rows = rows;
        *result_cols = cols;
        *new_shape_size = static_cast<int>(output_notation.size());

        *new_shape = new int[*new_shape_size];
        for (int i = 0; i < *new_shape_size; ++i)
        {
            (*new_shape)[i] = shape[i];
        }

        *result_data = new double[rows * cols];
        std::copy(data, data + rows * cols, *result_data);
        return;
    }

    // Split the input and output notation into UTF-8 characters
    std::vector<std::string> input_chars = split_utf8(input_notation);
    std::vector<std::string> output_chars = split_utf8(output_notation);

    // Convert shape array to vector for easier manipulation
    std::vector<int> shape_vec(shape, shape + input_chars.size());

    if (debug)
    {
        std::cout << "UTF-8 STRINGS:" << std::endl;
        std::cout << "Input Notation: ";
        for (const auto &ch : input_chars)
            std::cout << ch << ", ";
        std::cout << std::endl;
        std::cout << "Output Notation: ";
        for (const auto &ch : output_chars)
            std::cout << ch << ", ";
        std::cout << std::endl;
    }

    assert(input_notation.find(',') == std::string::npos);

    std::vector<int> dimensions;
    std::vector<Entry> entries;
    std::vector<int> sum_indices;
    std::vector<int> diag_indices;
    std::vector<int> perm_indices;

    // Handle diagonal indices
    find_diag_indices(input_chars, diag_indices, shape_vec);
    find_sum_indices(input_chars, output_chars, sum_indices, shape_vec);
    find_perm_indices(input_chars, output_chars, perm_indices, shape_vec);

    // Debug notation changes
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
    if (debug)
    {
        std::cout << "perm_indices: " << std::endl;
        for (auto i : perm_indices)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }

    // clock_t start, end;

    fill_dimensions_and_entries(data, rows, cols, dimensions, entries);

    // start = clock();
    if (diag_indices.size() != 0)
    {
        remove_non_diag_indices(dimensions, entries, diag_indices, cols);
    }
    // end = clock();
    // diagonal_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "diagonal_time: " << diagonal_time << "s" << std::endl;

    // start = clock();
    if (sum_indices.size() != 0)
    {
        sum_over_trivial_indices(dimensions, entries, sum_indices, cols);
    }
    // end = clock();
    // sum_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "sum_time: " << sum_time << "s" << std::endl;

    // // start = clock();
    if (perm_indices.size() != 0)
    {
        apply_permutation(dimensions, entries, perm_indices);
    }
    // end = clock();
    // permute_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "permute_time: " << permute_time << "s" << std::endl;

    // parallel_sort(entries, dimensions);

    // Prepare the output data
    *result_rows = static_cast<int>(entries.size());
    *result_cols = static_cast<int>(output_chars.size()) + 1;
    size_t shape_vec_size = shape_vec.size();

    if (shape_vec_size != 0)
    {
        *new_shape_size = static_cast<int>(shape_vec_size);
        *new_shape = new int[*new_shape_size];
        for (int i = 0; i < *new_shape_size; ++i)
        {
            (*new_shape)[i] = shape_vec[i];
        }
    }
    else
    {
        *result_cols = 2;
        dimensions.emplace_back(0);
        entries[0].num_idx = 1;
        *new_shape_size = 1;
        *new_shape = new int[*new_shape_size];
        (*new_shape)[0] = 1;
    }

    *result_data = new double[*result_rows * *result_cols];

    int r = 0;
    for (const auto &entry : entries)
    {
        for (size_t c = 0; c < entry.num_idx; ++c)
        {
            int value = dimensions[entry.offset + c] + 1;
            (*result_data)[r * *result_cols + c] = value - 1;
        }
        (*result_data)[r * *result_cols + (*result_cols - 1)] = entry.value;
        ++r;
    }
}

// Function to calculate flat index for a single set of indices
int ravel_single_index(const std::vector<int> &indices, const std::vector<int> &shape)
{
    if (indices.size() != shape.size())
    {
        throw std::invalid_argument("Indices and shape must have the same length");
    }
    int flat_index = 0;
    int stride = 1;
    // Iterate over the dimensions in reverse order
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }
    return flat_index;
}

void reshape(const double *data, int data_rows, int data_cols,
             const int *shape, const int shape_length,
             const int *new_shape, const int new_shape_length,
             double **result_data, int *result_rows, int *result_cols)
{
    // Check if the total number of elements is the same
    if (std::accumulate(shape, shape + shape_length, 1, std::multiplies<int>()) !=
        std::accumulate(new_shape, new_shape + new_shape_length, 1, std::multiplies<int>()))
    {
        throw std::invalid_argument("The total number of elements must remain the same for reshaping.");
    }

    // Convert shape and new_shape to vectors
    std::vector<int> shape_vec(shape, shape + shape_length);
    std::vector<int> new_shape_vec(new_shape, new_shape + new_shape_length);
    std::vector<double> coo_values(data_rows);
    std::vector<int> original_flat_indices;

    // TODO: tbb sorting works on this vector!!!!!!
    // tbb::parallel_sort(original_flat_indices.begin(), original_flat_indices.end());

    original_flat_indices.reserve(data_cols - 1);
    for (int i = 0; i < data_rows; ++i)
    {
        std::vector<int> indices(data_cols - 1);
        for (int j = 0; j < data_cols - 1; ++j)
        {
            indices[j] = static_cast<int>(data[i * data_cols + j]);
        }
        original_flat_indices.push_back(ravel_single_index(indices, shape_vec));
    }

    *result_rows = static_cast<int>(original_flat_indices.size());
    *result_cols = new_shape_length + 1;
    *result_data = new double[*result_rows * *result_cols];

    // Unravel and write directly to result_data
    for (int i = 0; i < *result_rows; ++i)
    {
        int flat_index = original_flat_indices[i];
        for (int j = static_cast<int>(new_shape_vec.size()) - 1; j >= 0; --j)
        {
            (*result_data)[i * *result_cols + j] = flat_index % new_shape_vec[j];
            flat_index /= new_shape_vec[j];
        }
        (*result_data)[i * *result_cols + (*result_cols - 1)] = data[i * data_cols + (data_cols - 1)];
    }
}

// Function to encode multi-dimensional indices into a single 64-bit integer
uint64_t encode_indices(const std::vector<int> &indices)
{
    uint64_t encoded_index = 0;
    int n = static_cast<int>(indices.size());
    for (int k = 0; k < n; ++k)
    {
        encoded_index |= (static_cast<uint64_t>(indices[k]) << k);
    }
    return encoded_index;
}

// Function to convert dense tensor data to encoded COO format for N-dimensional tensors with binary dimensions
void dense_to_encoded_coo(double *data, const int data_size, std::vector<uint64_t> &encoded_indices, std::vector<double> &coo_values)
{
    int total_elements = 1 << data_size; // 2^n where n is the number of dimensions

    for (int i = 0; i < total_elements; ++i)
    {
        double value = data[i];
        if (value != 0.0)
        {
            std::vector<int> indices(data_size);
            int idx = i;
            for (int j = data_size - 1; j >= 0; --j)
            {
                indices[j] = idx % 2; // Each dimension is binary (0 or 1)
                idx /= 2;
            }
            uint64_t encoded_index = encode_indices(indices);
            encoded_indices.push_back(encoded_index);
            coo_values.push_back(value);
        }
    }
}

void einsum_dim_2(
    uint32_t *in_out_flat,
    int32_t *in_out_sizes,
    int n_tensors,
    int n_map_items,
    uint32_t *keys_sizes,
    uint64_t *values_sizes,
    int32_t *path,
    double *data)
{
    int32_t n_idx = std::accumulate(in_out_sizes, in_out_sizes + n_tensors, 0, std::plus<int32_t>());
    std::unordered_map<int, int> sizes;
    std::vector<double *> data_ptr;

    for (int i = 0; i < n_map_items; ++i)
    {
        sizes[keys_sizes[i]] = static_cast<uint32_t>(values_sizes[i]);
    }

    // Get pointers to tensors in data
    double *tmp = data;
    for (int i = 0; i < n_tensors; ++i)
    {
        data_ptr.emplace_back(tmp);
        tmp += (1ULL << in_out_sizes[i]);
    }

    for (int i = 0; i < n_tensors - 1; ++i)
    {
        int idx_size = in_out_sizes[i];
        int data_size = (1 << idx_size);

        std::vector<uint64_t> encoded_indices;
        std::vector<double> coo_values;

        dense_to_encoded_coo(data_ptr[i], data_size, encoded_indices, coo_values);

        // TODO: Adjust pointers to point to the encoded data and use them for
        // further computations

        // Print the encoded COO format
        std::cout << "Tensor " << i << " in encoded COO format:" << std::endl;
        for (size_t j = 0; j < coo_values.size(); ++j)
        {
            std::cout << "Index: " << encoded_indices[j] << ", Value: " << coo_values[j] << std::endl;
        }
    }

    for (int i = 0; i < n_idx; ++i)
    {
        std::cout << in_out_flat[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < n_tensors; ++i)
    {
        std::cout << in_out_sizes[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < n_map_items; ++i)
    {
        std::cout << keys_sizes[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < n_map_items; ++i)
    {
        std::cout << values_sizes[i] << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < (n_tensors - 2) * 2; i += 2)
    {
        std::cout << "(" << path[i] << ", " << path[i + 1] << "), ";
    }
    std::cout << std::endl;

    for (int i = 0; i < (n_tensors - 2) * 2; i += 2)
    {
        int32_t t_1 = path[i];
        int32_t t_2 = path[i + 1];
        std::cout << t_1 << ", " << t_2 << ", " << n_tensors << std::endl;
        // Tensor data
        double *data_1 = data_ptr[t_1];
        double *data_2 = data_ptr[t_2];
        double *data_out = data_ptr[n_tensors - 1];
        std::cout << "Tensors:" << std::endl;
        std::cout << data_1[0] << ", " << data_1[1] << std::endl;
        std::cout << data_1[2] << ", " << data_1[3] << std::endl;
        std::cout << data_2[0] << ", " << data_2[1] << std::endl;
        std::cout << data_2[2] << ", " << data_2[3] << std::endl;
        std::cout << data_out[0] << ", " << data_out[1] << std::endl;
        std::cout << data_out[2] << ", " << data_out[3] << std::endl
                  << std::endl;

        int32_t t_1_size = in_out_sizes[t_1];
        int32_t t_2_size = in_out_sizes[t_2];
        int32_t t_out_size = in_out_sizes[n_tensors - 1];

        int32_t in_out_offset_1 = std::accumulate(in_out_sizes, in_out_sizes + t_1, 0, std::plus<int32_t>());
        int32_t in_out_offset_2 = std::accumulate(in_out_sizes, in_out_sizes + t_2, 0, std::plus<int32_t>());
        int32_t in_out_offset_out = std::accumulate(in_out_sizes, in_out_sizes + n_tensors - 1, 0, std::plus<int32_t>());

        uint32_t *t_2_begin = in_out_flat + in_out_offset_2;
        uint32_t *t_2_end = in_out_flat + in_out_offset_2 + t_2_size;
        uint32_t *t_out_begin = in_out_flat + in_out_offset_out;
        uint32_t *t_out_end = in_out_flat + in_out_offset_out + t_out_size;

        std::unordered_set<uint32_t> seen;
        std::vector<uint32_t> batch_idx;
        std::vector<uint32_t> keep_idx;
        std::vector<uint32_t> sum_idx;
        std::vector<uint32_t> contract_idx;

        for (int x = 0; x < t_1_size; ++x)
        {
            uint32_t id_1 = in_out_flat[in_out_offset_1 + x];
            uint32_t id_2 = in_out_flat[in_out_offset_2 + x];

            if (seen.find(id_1) != seen.end())
                continue;
            seen.emplace(id_1);

            if (t_2_end != std::find(t_2_begin, t_2_end, id_1))
            {
                if (t_out_end != std::find(t_out_begin, t_out_end, id_1))
                {
                    batch_idx.emplace_back(id_1);
                }
                else
                {
                    contract_idx.emplace_back(id_1);
                }
            }
            else if (t_out_end != std::find(t_out_begin, t_out_end, id_1))
            {
                keep_idx.emplace_back(id_1);
            }
            else
            {
                sum_idx.emplace_back(id_1);
            }
        }

        std::cout << "BATCH: ";
        for (auto i : batch_idx)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "CON: ";
        for (auto i : contract_idx)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "KEEP: ";
        for (auto i : keep_idx)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
        std::cout << "SUM: ";
        for (auto i : sum_idx)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;

        // TODO: Find indices indicating a diagonal computation.
        // Afterwards perform diagonal computation, trivial sum, and batch/con
        // solving. Finally adjust to the new shape
    }
}

//////// LEGACY FUNCTIONS ////////

double cpu_time_used_legacy = 0;

void coo_bmm_legacy(const double *A_data, int A_rows, int A_cols,
                    const double *B_data, int B_rows, int B_cols,
                    double **C_data, int *C_rows, int *C_cols)
{
    // clock_t start, end;
    // double cpu_time_used_legacy = 0;
    // start = clock();
    // end = clock();
    // cpu_time_used_legacy += ((double)(end - start)) / CLOCKS_PER_SEC;
    // std::cout << "BMM Time: " << cpu_time_used_legacy << "s" << std::endl;

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
void find_sum_indices_legacy(std::string &input_notation, const std::string &output_notation,
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
void find_diag_indices_legacy(std::string &input_notation, std::vector<int> &diag_indices)
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
void find_perm_indices_legacy(const std::string &input_notation, const std::string &output_notation,
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

void sum_over_trivial_indices_legacy(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
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

void remove_non_diag_indices_legacy(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
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

void apply_permutation_legacy(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
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

void result_map_to_data_legacy(std::unordered_map<std::vector<int>, double, VectorHash, VectorEqual> &result_map,
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

void single_einsum_legacy(const double *data, int rows, int cols,
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
    find_diag_indices_legacy(input_notation, diag_indices);

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

    remove_non_diag_indices_legacy(result_map, diag_indices);

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

    find_sum_indices_legacy(input_notation, output_notation, sum_indices);

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

    sum_over_trivial_indices_legacy(result_map, sum_indices);

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

    find_perm_indices_legacy(input_notation, output_notation, perm_indices);

    if (debug)
    {
        std::cout << "perm_indices: " << std::endl;
        for (auto i : perm_indices)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }

    apply_permutation_legacy(result_map, perm_indices);

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
/* Function declarations */

#ifndef GENERATE_SAMPLES_H
#define GENERATE_SAMPLES_H

// Function to generate a sample table along with the (absolute) minus log probability that we sampled that table
std::tuple<std::vector<std::vector<int>>, double> sample_table(std::vector<int> ks, int m_in);
// Sample the diagonal sum of a symmetric table with given margin
int sample_diagonal_sum(std::vector<int> ks);
// Sample the diagonal entries of a symmetric table with given margin and diagonal sum
std::vector<int> sample_diagonal_entries(std::vector<int> ks, int m_in);
// Sample the off-diagonal entries of the table given the remaining margin (passing the table by reference to be written)
void sample_off_diagonal_table(std::vector<std::vector<int>>& table, std::vector<int>& ks, int m_in);
// Sample the rows one at a time (passing the table by reference to be written as well as the margin ks to be updated)
void sample_table_row(std::vector<std::vector<int>>& table, std::vector<int>& ks, int i);

#endif // GENERATE_SAMPLES_H
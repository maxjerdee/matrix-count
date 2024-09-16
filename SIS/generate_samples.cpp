// Example usage: ./generate_samples -i example.txt -T 1


#include <getopt.h>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "globals.h"
#include "preprocessing.h"
#include "generate_samples.h"
#include "estimates.h"
#include "helpers.h"

// Main program
int main(int argc, char *argv[]) {
    // Get filenames from the command line input (and update global options)
    auto [input_filename, output_filename] = get_options(argc, argv);

    // Read margins (and diagonal sum) specification from input_filename. m_in = -1 indicates an unconstrained diagonal sum
    auto [ks, m_in] = read_data(input_filename);

    // Check whether there are any solutions to the input margins with the given diagonal sum
    validate_input(ks, m_in);

    print(ks);
    print(m_in);

    // Set the random seed
    // std::srand(std::time(0));
    std::srand(1);

    // Initialize the transition probability tables given the appropriate size, this array will be reused for various purposes, although we only allocate the array in memory once
    double ***log_g_i_sP_sT; // log_g_i_sP_sT[i][s_prev][s_this] is log g_{i+1}(s_prev,s_this)
    log_g_i_sP_sT = new double **[n];
    for(int i = 0; i < n; i++){
      log_g_i_sP_sT[i] = new double *[m+1];
      for(int s_prev = 0; s_prev < m+1; s_prev++){
        log_g_i_sP_sT[i][s_prev] = new double[m+1];
        for(int s_this = 0; s_this < m+1; s_this++){
          log_g_i_sP_sT[i][s_prev][s_this] = 0;
        }
      }
    }

    // Start timer for sampling
    auto start = std::chrono::high_resolution_clock::now();
    // Generate samples until the maximum number of iterations or time elpased is reached
    for(int t = 0; t < max_iterations; t++){
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        if(duration.count() >= max_time){ // Stop loop when maximum time has elapsed
            break;
        }
        // Generate a sample table along with the (absolute) minus log probability that we sampled that table
        auto [table, minus_log_prob] = sample_table(ks, m_in);
        print(table);
    }
}

// Function to generate a sample table along with the (absolute) minus log probability that we sampled that table
std::tuple<std::vector<std::vector<int>>, double> sample_table(std::vector<int> ks, int m_in){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    // Initialize the table and the minus log probability
    std::vector<std::vector<int>> table; // Table to be returned
    for(int i = 0; i < n; i++){
        std::vector<int> row(n, 0);
        table.push_back(row);
    }
    double minus_log_prob = 0;

    if(m_in == -1){ // Unconstrained diagonal sum
        // Replace with an appropriately sampled value
        m_in = sample_diagonal_sum(ks);
    }

    // Sample the diagonal entries of the table among the possibilities with the appropriate total
    std::vector<int> diagonal_entries = sample_diagonal_entries(ks, m_in);
    // Write the diagonal entries to the table and adjust the remaining margin
    for(int i = 0; i < n; i++){
        table[i][i] = diagonal_entries[i];
        ks[i] -= diagonal_entries[i];
    }

    // Sample the off-diagonal entries of the table given the remaining margin (passing the table by reference to be written)
    sample_off_diagonal_table(table, ks, m_in);

    // Return the table and the minus log probability
    return std::make_tuple(table, minus_log_prob);
}

// Sample the diagonal sum of a symmetric table with given margin
int sample_diagonal_sum(std::vector<int> ks){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    // Calculate the bounds of possible diagonal sums
    // Minimum allowable diagonal sum
    int ks_max = *std::max_element(ks.begin(), ks.end());
    int m_in_min = ks_max - m;
    if(m_in_min < 0){ // Also must be at least 0
        m_in_min = 0;
    }

    // Maximum allowable diagonal sum
    int m_in_max = 0;
    for(int k : ks){
        m_in_max += std::floor(k/2);
    }

    // Get the (log) counts for the number of possible matrices for the valid diagonal sums
    std::vector<double> log_counts;
    for(int m_in = m_in_min; m_in <= m_in_max; m_in++){
        log_counts.push_back(log_Omega_fixed_diagonal(ks, m_in));
    }

    int m_in_index = sample_log_weights(log_counts); // Sample the index of the log_counts
    return m_in_index + m_in_min; // Return the corresponding diagonal sum
}

// Sample the diagonal entries of a symmetric table with given margin and diagonal sum
std::vector<int> sample_diagonal_entries(std::vector<int> ks, int m_in){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    int m_out = m - m_in; // Number of off-diagonal edges

    // Sample the diagonal entries
    std::vector<int> ds;
    // These are going to be weighted by the number of symmetric matrices that can fill the off-diagonal entries (i.e. zero digaonal sum)
    double alpha = alpha_zero_diagonal(n, m_out); // The relevant alpha for the approximation
    // h_i(s_prev, s_this) = binom(ks[i]-2 ds[i] + alpha - 1, alpha - 1)*1{0,(ks[i]-m_out)/2<=ds[i]<=ks[i]/2}
    // The hard constraints should just be imposed by the bounds of the loops that compute the sums that make up the g_i(s_prev,s_this)
    int d_min_next;
    int d_max_next;
    for(int i = n; i > 0; i--){
        int d_min = std::max(0, (ks[i] - m_out)/2);
        int d_max = ks[i]/2;
        if(i == n){ // Should just be equal to the log_h
            for(int d = d_min; d <= d_max; d++){
                log_g_i_sP_sT[i - 1][m_in - d][m_in] = log_binom(ks[i] - 2*d + alpha - 1, alpha - 1);
            }
        }else{
            for(int d = d_min; d <= d_max; d++){
                
            }
        }
        d_min_next = d_min; // Store these in order to understand what the bounds are
        d_max_next = d_max;
    }
    // TODO: check in mathematica that things are working and then map out what the appropriate ranges are to calculate the g values. 

    return ds;
}

// Sample the off-diagonal entries of the table given the remaining margin (passing the table by reference to be written)
void sample_off_diagonal_table(std::vector<std::vector<int>>& table, std::vector<int>& ks, int m_in){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    // Sample the rows one at a time (passing the table by reference to be written as well as the margin ks to be updated)
    for(int i = 0; i < n; i++){
        sample_table_row(table, ks, i);
    }
}

// Sample the rows one at a time (passing the table by reference to be written as well as the margin ks to be updated)
void sample_table_row(std::vector<std::vector<int>>& table, std::vector<int>& ks, int i){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    // Sample the row
    for(int j = i + 1; j < n; j++){
        table[i][j] = std::rand() % (std::min(ks[i], ks[j]) + 1); // Placeholder, TODO: change to the actual weighted sampling
        table[j][i] = table[i][j]; // Symmetric table
        ks[i] -= table[i][j];
        ks[j] -= table[i][j];
    }
}




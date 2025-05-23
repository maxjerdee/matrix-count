// Example usage: ./generate_samples -i example.txt -T 1


#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>

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

    printf("ks = ");
    print(ks);
    printf("m_in = %d\n", m_in);

    // Compute the linear time estimate
    double log_Omega;
    if(m_in == -1){
        log_Omega = log_Omega_unconstrained_diagonal(ks);
        printf("Linear time estimate: %f\n", log_Omega);
    }else{
        log_Omega = log_Omega_fixed_diagonal(ks, m_in);
        printf("Linear time estimate: %f\n", log_Omega);
    }

    // Extract relevant metadata
    n = ks.size(); // Number of rows/columns, globally defined
    m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges, globally defined

    // Set the random seed
    std::srand(std::time(0));
    // std::srand(10);

    // Initialize the transition probability tables given the appropriate size, this array will be reused for various purposes, although we only allocate the array in memory once
    // log_g_i_sP_sT[i][s_prev][s_this] is log g_{i+1}(s_prev,s_this)
    log_g_i_sP_sT = new double **[n]; // Globally defined
    for(int i = 0; i < n; i++){
      log_g_i_sP_sT[i] = new double *[m+1];
      for(int s_prev = 0; s_prev < m+1; s_prev++){
        log_g_i_sP_sT[i][s_prev] = new double[m+1];
        for(int s_this = 0; s_this < m+1; s_this++){
          log_g_i_sP_sT[i][s_prev][s_this] = 0;
        }
      }
    }

    
    // Open output file, clear and then append live
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "Error: unable to open file " << output_filename << std::endl;
        exit(0);
    }
    outfile << std::unitbuf; // Make sure the output is flushed after each write

    outfile << "log_Omega = " << log_Omega << std::endl;

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
        auto [table, entropy] = sample_table(ks, m_in);
        // print("table:");
        // print(table);
        // printf("entropy = %f\n", entropy);

        // // Write the table
        // for(int i = 0; i < n; i++){
        //     for(int j = 0; j < n; j++){
        //         outfile << table[i][j] << " ";
        //     }
        // }

        // int sampled_m_in = 0;
        // for(int i = 0; i < n; i++){
        //     sampled_m_in += table[i][i]/2;
        // }
        // // Write m_in
        // outfile << sampled_m_in << " ";

        // Write the entropy to the output file
        outfile << entropy << std::endl;
        outfile.flush();
    }
}

// Function to generate a sample table along with the (absolute) minus log probability that we sampled that table
std::pair<std::vector<std::vector<int>>, double> sample_table(std::vector<int> ks, int m_in){
    // Initialize the table and the minus log probability
    std::vector<std::vector<int>> table; // Table to be returned
    for(int i = 0; i < n; i++){
        std::vector<int> row(n, 0);
        table.push_back(row);
    }
    double entropy = 0;

    if(m_in == -1){ // Unconstrained diagonal sum
        // Replace with an appropriately sampled value
        auto [m_in_sampled, diagonal_sum_entropy] = sample_diagonal_sum(ks);
        m_in = m_in_sampled;
        entropy += diagonal_sum_entropy;
        // printf("Sampled m_in = %d, entropy = %f\n", m_in, diagonal_sum_entropy);
    }

    // Sample the diagonal entries of the table among the possibilities with the appropriate total
    auto [diagonal_entries, diagonal_entries_entropy] = sample_diagonal_entries(ks, m_in);
    entropy += diagonal_entries_entropy;

    // printf("Sampled diagonal entries, entropy = %f\n", diagonal_entries_entropy);
    // print(diagonal_entries);
    
    // Write the diagonal entries to the table and adjust the remaining margin
    for(int i = 0; i < n; i++){
        table[i][i] = 2*diagonal_entries[i];
    }

    // Sample the off-diagonal entries of the table given the remaining margin (passing the table by reference to be written)
    double off_diag_entropy = sample_off_diagonal_table(table, ks, m_in);
    entropy += off_diag_entropy;

    // Return the table and the minus log probability
    return std::make_pair(table, entropy);
}

// Sample the diagonal sum of a symmetric table with given margin (and return the entropy that sample)
std::pair<int,double> sample_diagonal_sum(std::vector<int> ks){
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

    auto [m_in_index, entropy] = sample_log_weights(log_counts); // Sample the index of the log_counts
    // print(entropy);
    return std::make_pair(m_in_index + m_in_min,entropy); // Return the corresponding diagonal sum
}

// Sample the diagonal entries of a symmetric table with given margin and diagonal sum (and give entropy of that choice)
std::pair<std::vector<int>,double> sample_diagonal_entries(std::vector<int> ks, int m_in){
    int m_out = m - m_in; // Number of off-diagonal edges
    if(m_in < 0){
        print("Error: m_in negative");
        exit(1);
    }
    if(m_out < 0){
        print("Error: m_out negative");
        exit(1);
    }
    
    // Use dynamic programming to compute the log_g values
    // These are going to be weighted by the number of symmetric matrices that can fill the off-diagonal entries (i.e. zero digaonal sum)
    double alpha = alpha_zero_diagonal(n, m_out) + ALPHA_EPSILON; // The relevant alpha for the approximation    
    // printf("alpha = %f\n", alpha);
    // printf("n = %d, m_out = %d\n", n, m_out);
    // h_i(s_prev, s_this) = binom(ks[i]-2 ds[i] + alpha - 1, alpha - 1)*1{0,(ks[i]-m_out)/2<=ds[i]<=ks[i]/2}
    // The hard constraints should just be imposed by the bounds of the loops that compute the sums that make up the g_i(s_prev,s_this)
    int d_min_next;
    int d_max_next;
    std::vector<int> min_s_this = std::vector<int>(n + 1, 10000); // Track the smallest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
    std::vector<int> max_s_this = std::vector<int>(n + 1, -1); // Track the largest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
    min_s_this[n] = m_in;
    max_s_this[n] = m_in;
    for(int i = n - 1; i >= 0; i--){
        int d_min = std::max(0.0,std::ceil(static_cast<double>(ks[i] - m_out)/2));
        int d_max = std::floor(static_cast<double>(ks[i])/2);
        if(i == n - 1){ // Should just be equal to the log_h
            for(int d_this = d_min; d_this <= std::min(d_max,m_in); d_this++){ // The range of valid d[i], writing in the nonzero log g_n(s_{i-1},s_i), make sure s_prev >= 0
                int s_prev = m_in - d_this; // s_{i-1}
                int s_this = m_in; // s_i
                log_g_i_sP_sT[i][s_prev][s_this] = log_binom(ks[i] - 2*d_this + alpha - 1, alpha - 1);\
            }
            min_s_this[i] = std::max(0,m_in - d_max); // The smallest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
            max_s_this[i] = m_in - d_min; // The largest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
        }else{
            // Range of valid d[i+1], for finding the contributing g_{i+1}(s_i,s_{i+1})
            int d_next_min = std::max(0.0,std::ceil(static_cast<double>(ks[i+1] - m_out)/2));
            int d_next_max = std::floor(static_cast<double>(ks[i+1])/2);
            // g_i(s_{i-1},s_i) = h_i(s_{i-1},s_i)*\sum_{s_i} g_{i+1}(s_i,s_{i+1})
            for(int s_this = min_s_this[i+1]; s_this <= max_s_this[i+1]; s_this++){ // Writing in the nonzero log g_n(s_{i-1},s_i)
                for(int d_this = d_min; d_this <= std::min(d_max,s_this); d_this++){ // Also ensure that s_{i-1} >= 0
                    int s_prev = s_this - d_this; // s_{i-1}
                    if(i != 0 || s_prev == 0){ // The first s_{i-1} is always 0
                        double log_h = log_binom(ks[i] - 2*d_this + alpha - 1, alpha - 1);
                        std::vector<double> log_gs; // The log(g) values that the next g is defined as a sum over.
                        for(int d_next = d_next_min; d_next <= std::min(d_next_max,m_in - s_this); d_next++){ // Make sure that s_{i+1} <= m_in
                            int s_next = s_this + d_next; // s_{i+1}
                            if(i != n - 2 || s_next == m_in){ // The last s_{i+1} is always m_in
                                log_gs.push_back(log_g_i_sP_sT[i+1][s_this][s_next]);
                            }
                        }
                        log_g_i_sP_sT[i][s_prev][s_this] = log_h + log_sum_exp(log_gs);
                        // printf("log_g[%d][%d][%d] = %f\n", i, s_prev, s_this, log_g_i_sP_sT[i][s_prev][s_this]);
                        // Update the min_s_this and max_s_this
                        if(s_prev < min_s_this[i]){
                            min_s_this[i] = s_prev;
                        }
                        if(s_prev > max_s_this[i]){
                            max_s_this[i] = s_prev;
                        }
                    }
                }
            }
        }
    }
    // print(min_s_this);
    // print(max_s_this);
    // Sample from the distribution given by the g
    std::vector<int> ds;
    double entropy = 0;
    int s_prev = 0;
    for(int i = 0; i < n; i++){
        // printf("i = %d\n", i);
        int d_min = std::max(0.0,std::ceil(static_cast<double>(ks[i] - m_out)/2));
        int d_max = std::floor(static_cast<double>(ks[i])/2);
        std::vector<double> weights; // Weights given by the valid weights
        std::vector<int> d_choices; // Choices of degree
        for(int d_this = d_min; d_this <= std::min(d_max,m_in - s_prev); d_this++){ // Make sure that s_{i} <= m_in
            int s_this = s_prev + d_this;
            if(s_this >= min_s_this[i+1] && s_this <= max_s_this[i+1]){ // Make sure that the resulting s_this is valid
                if(i != n - 1 || s_this == m_in){
                    // printf("s_prev = %d, s_this = %d, log_g = %f\n", s_prev, s_this, log_g_i_sP_sT[i][s_prev][s_this]);
                    weights.push_back(log_g_i_sP_sT[i][s_prev][s_this]);
                    d_choices.push_back(d_this);
                }
            }
        }
        auto [d_this_index, d_entropy] = sample_log_weights(weights);
        entropy += d_entropy;
        int d_this = d_choices[d_this_index];
        ds.push_back(d_this);
        s_prev += d_this;
    }

    return std::make_pair(ds,entropy);
}

// Sample the off-diagonal entries of the table given the remaining margin (passing the table by reference to be written)
double sample_off_diagonal_table(std::vector<std::vector<int>>& table, std::vector<int>& ks, int m_in){
    // Sample the rows one at a time (passing the table by reference to be written as well as the margin ks to be updated)
    double entropy = 0;
    // Make a copy of the degrees to be decremented as we sample
    std::vector<int> ks_left;
    for(int i = 0; i < n; i++){
        ks_left.push_back(ks[i] - table[i][i]);
    }
    std::vector<int> is; // Which indices in the table are represented by the remaining degrees (allows us to remove them as 0s appear and rows are sampled)
    for(int i = 0; i < n; i++){
        is.push_back(i);
    }
    for(int i = 0; i < n; i++){
        // Re-initialize the log table
        // TODO: Fix the code so that we don't need to allocate this memory every time.. 
        for(int i = 0; i < n; i++){
            for(int s_prev = 0; s_prev < m+1; s_prev++){
                for(int s_this = 0; s_this < m+1; s_this++){
                    // -inf
                log_g_i_sP_sT[i][s_prev][s_this] = -std::numeric_limits<double>::infinity();
                }
            }
        }
        entropy += sample_table_row(table, ks_left, is);
        // exit(0);
    }
    return entropy;
}

// Sample the rows one at a time (passing the table by reference to be written as well as the margin ks to be updated)
double sample_table_row(std::vector<std::vector<int>>& table, std::vector<int>& ks, std::vector<int>& is){
    // print("ks:");
    // print(ks);
    // Remove the rows that have degree 0
    std::vector<int> k_zero_inds;
    for(int i = 0; i < is.size(); i++){
        if(ks[i] == 0){
            k_zero_inds.push_back(i);
        }
    }
    for(int i = k_zero_inds.size() - 1; i >= 0; i--){
        int ind = k_zero_inds[i];
        is.erase(is.begin() + ind);
        ks.erase(ks.begin() + ind);
    }
    // print(ks);
    // print(is);
    int n_left = ks.size(); // Number of remaining rows
    int m_out = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of remaining off-diagonal edges
    if(n_left == 0){
        return 0;
    }
    // Sample the top row, assuming that all of the diagonal values are 0 (a length n_left - 1 vector that sums to ks[0])
    // Use dynamic programming to compute the log_g values
    // If the number left is 4 or less, we can set alpha = 1 since all remaining configurations are equally likely (single solution)
    double alpha;
    if(n_left <= 4){
        alpha = 1;
    }else{
        alpha = alpha_zero_diagonal(n_left - 1, m_out - 2*ks[0]) + ALPHA_EPSILON; // The relevant alpha for the approximation
    }    
    // if(n_left == 6){
    //     print("alpha:");
    //     print(alpha);
    // }


    // h_i(s_prev, s_this) = binom(ks[i]- as[i] + alpha - 1, alpha - 1)*1{0,ks[i]+ks[1]-m<=as[i]<=ks[i]}
    // The hard constraints should just be imposed by the bounds of the loops that compute the sums that make up the g_i(s_prev,s_this)
    int a_min_next;
    int a_max_next;
    std::vector<int> min_s_this = std::vector<int>(n_left, 10000); // Track the smallest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
    std::vector<int> max_s_this = std::vector<int>(n_left, -1); // Track the largest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
    min_s_this[n_left - 1] = ks[0];
    max_s_this[n_left - 1] = ks[0];
    for(int i = n_left - 1 - 1; i >= 0; i--){
        int a_min = std::max(0,ks[i+1] + ks[0] - m_out);
        int a_max = ks[i+1];
        if(i == n_left - 1 - 1){ // Should just be equal to the log_h
            for(int a_this = a_min; a_this <= std::min(a_max,ks[0]); a_this++){ // The range of valid a[i], writing in the nonzero log g_n(s_{i-1},s_i), make sure s_prev >= 0
                int s_prev = ks[0] - a_this; // s_{i-1}
                int s_this = ks[0]; // s_i
                log_g_i_sP_sT[i][s_prev][s_this] = log_binom(ks[i + 1] - a_this + alpha - 1, alpha - 1); // log h_i(s_{i-1},s_i)
                // if(n_left == 6){
                //     printf("log_g[%d][%d][%d] = %f\n", i, s_prev, s_this, log_g_i_sP_sT[i][s_prev][s_this]);
                // }
            }
            min_s_this[i] = std::max(0,ks[0] - a_max); // The smallest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
            max_s_this[i] = ks[0] - a_min; // The largest s_i with nonzero \sum_{s_{i+1}} g_i(s_i, s_{i+1})
        }else{
            // Range of valid d[i+1], for finding the contributing g_{i+1}(s_i,s_{i+1})
            int a_next_min = std::max(0,ks[i+2] + ks[0] - m_out);
            int a_next_max = ks[i+2];
            // g_i(s_{i-1},s_i) = h_i(s_{i-1},s_i)*\sum_{s_i} g_{i+1}(s_i,s_{i+1})
            for(int s_this = min_s_this[i+1]; s_this <= max_s_this[i+1]; s_this++){ // Writing in the nonzero log g_n(s_{i-1},s_i)
                for(int a_this = a_min; a_this <= std::min(a_max,s_this); a_this++){ // Also ensure that s_{i-1} >= 0
                    int s_prev = s_this - a_this; // s_{i-1}
                    if(i != 0 || s_prev == 0){ // The first s_{i-1} is always 0
                        double log_h = log_binom(ks[i + 1] - a_this + alpha - 1, alpha - 1); // log h_i(s_{i-1},s_i)
                        std::vector<double> log_gs; // The log(g) values that the next g is defined as a sum over.
                        for(int a_next = a_next_min; a_next <= std::min(a_next_max,ks[0] - s_this); a_next++){ // Make sure that s_{i+1} <= m_in
                            int s_next = s_this + a_next; // s_{i+1}
                            if(i != n_left - 1 - 1 - 1 || s_next == ks[0]){ // The last s_{i+1} is always ks[0]
                                log_gs.push_back(log_g_i_sP_sT[i+1][s_this][s_next]);
                            }
                        }
                        log_g_i_sP_sT[i][s_prev][s_this] = log_h + log_sum_exp(log_gs);
                        // if(n_left == 6){
                        //     print(log_gs);
                        //     // printf("ks[i+1] = %d, a_this = %d\n", ks[i+1], a_this);
                        //     printf("log_binom(%f,%f)\n",ks[i+1] - a_this + alpha - 1,alpha - 1);
                        //     print(log_h);
                        //     printf("log_g[%d][%d][%d] = %f\n", i, s_prev, s_this, log_g_i_sP_sT[i][s_prev][s_this]);
                        // }
                        // Update the min_s_this and max_s_this
                        if(s_prev < min_s_this[i]){
                            min_s_this[i] = s_prev;
                        }
                        if(s_prev > max_s_this[i]){
                            max_s_this[i] = s_prev;
                        }
                    }
                }
            }
        }
    }
    // print("min_s_this");
    // print(min_s_this);
    // print(max_s_this);
    // Sample from the distribution given by the g
    std::vector<int> as;
    double entropy = 0;
    int s_prev = 0;
    // print(n_left - 1);
    for(int i = 0; i < n_left - 1; i++){
        // printf("i = %d\n", i);
        int a_min = std::max(0,ks[i+1] + ks[0] - m_out);
        int a_max = ks[i+1];
        // printf("a_min = %d, a_max = %d\n", a_min, a_max);
        std::vector<double> weights; // Weights given by the valid weights
        std::vector<int> a_choices; // Choices of degree
        for(int a_this = a_min; a_this <= std::min(a_max,ks[0] - s_prev); a_this++){ // Make sure that s_{i} <= ks[0]
            int s_this = s_prev + a_this;
            if(s_this >= min_s_this[i+1] && s_this <= max_s_this[i+1]){ // Make sure that the resulting s_this is valid
                if(i != n_left - 1 - 1 || s_this == ks[0]){
                    // printf("i = %d, s_prev = %d, s_this = %d\n", i, s_prev, s_this);
                    weights.push_back(log_g_i_sP_sT[i][s_prev][s_this]);
                    a_choices.push_back(a_this);
                }
            }
        }
        // print("weights");
        // print(weights);
        // print(a_choices);
        auto [a_this_index, d_entropy] = sample_log_weights(weights);
        // print(d_entropy);
        entropy += d_entropy;
        // print(entropy);
        int a_this = a_choices[a_this_index];
        as.push_back(a_this);
        s_prev += a_this;
    }
    // print("as:");
    // print(as);
    // print("entropy:");
    // print(entropy);

    // Write the sampled row to the table and update the degrees
    int i = n - n_left; // Number of rows already sampled
    for(int k_i = 0; k_i < n_left - 1; k_i++){
        table[is[0]][is[k_i + 1]] = as[k_i];
        table[is[k_i + 1]][is[0]] = table[is[0]][is[k_i + 1]]; // Symmetric table
        ks[0] -= table[is[0]][is[k_i + 1]]; // Update the (shifted) margin
        ks[k_i + 1] -= table[is[k_i + 1]][is[0]]; // Update the (shifted) margin
    }
    // Remove the first element of ks and is
    ks.erase(ks.begin());
    is.erase(is.begin());
    return entropy;
}




/* Sequential importance sampling of non-negative integer matrices given conditions
 * 
 * Written by Max Jerdee  09 MAY 2024
 */

// After making the file, run the executable as ./DCSBM_old_resolution -i ../../data/networks/polbooks.gml -o ../../data/samples/polbooks.csv
// ./DCSBM_old_resolution -i ../../data/networks/polbooks.gml -o out.csv
// ./DCSBM_old_resolution -i ../../data/networks/dogs.gml -o out.csv -D t -M t -T t

#include <getopt.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <vector>
#include <iterator>
#include <random>
#include <vector>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <unordered_set>

/* Global variables */



/* Function declarations */

// Helper functions for computation
// Function to compute log(binomial(a,b))
double log_binom(double a, double b);
// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);

double log_factorial(int n);
double log_factorial2(int n);

// Write samples to file

// Printing quantities (for debugging)
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);

//*********************** MAIN PROGRAM ***********************************************************************

int main(int argc, char *argv[]) {
    // Parse command line parameters
    int opt;
    std::string input_filename;
    std::string output_filename;
    // take command line arguments
    while ((opt = getopt(argc, argv, "i:o:D:M:T:")) != -1) {
        switch (opt) {
        case 'i':
            input_filename = optarg;
            break;
        case 'o':
            output_filename = optarg;
            break;
        // case 'D':
        //     if(optarg[0] == 't'){
        //         CONDITION_ON_DIAGONAL = true;
        //     }else if(optarg[0] == 'f'){
        //         CONDITION_ON_DIAGONAL = false;
        //     }else{
        //         std::cout << "Error: invalid argument for -D flag (CONDITION_ON_DIAGONAL). Use 't' for true or 'f' for false.\n";
        //         exit(0);
        //     }
        //     break;
        // case 'M':
        //     if(optarg[0] == 't'){
        //         MULTIGRAPH_ENCODING = true;
        //     }else if(optarg[0] == 'f'){
        //         MULTIGRAPH_ENCODING = false;
        //     }else{
        //         std::cout << "Error: invalid argument for -M flag (MULTIGRAPH_ENCODING). Use 't' for true or 'f' for false.\n";
        //         exit(0);
        //     }
        //     break;
        // case 'T':
        //     if(optarg[0] == 't'){
        //         NEW_TRANSMISSION_ORDER = true;
        //     }else if(optarg[0] == 'f'){
        //         NEW_TRANSMISSION_ORDER = false;
        //     }else{
        //         std::cout << "Error: invalid argument for -T flag (NEW_TRANSMISSION_ORDER). Use 't' for true or 'f' for false.\n";
        //         exit(0);
        //     }
        //     break;
        default:
            exit(EXIT_FAILURE);
        }
    }
}


double log_binom(double a, double b){
    return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
}

double log_factorial(int n){
    return std::lgamma(n + 1);
}

double log_factorial2(int n){
    // Check that n is an even integer
    if(n % 2 != 0){
        std::cout << "Error: log_factorial2 called with odd integer.\n";
        exit(0);
    }
    return n/2*std::log(2) + std::lgamma(n/2 + 1);
}

double log_sum_exp(double a, double b){
    double a_b_max = std::max(a,b);
    return a_b_max + std::log(std::exp(a - a_b_max) + std::exp(b - a_b_max));
}


// Printing functions (useful for debugging)
void print(std::vector<std::vector<int>> table) {
  for(int i = 0; i < table.size(); i++){
    for(int j = 0; j < table[0].size(); j++){
        std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<std::vector<double>> table) {
  for(int i = 0; i < table.size(); i++){
    for(int j = 0; j < table[0].size(); j++){
        std::cout << table[i][j] << ' ';
    }
    std::cout << "\n";
  }
}
void print(std::vector<int> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(std::vector<double> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print(int val){
  std::cout<<val<<std::endl;
}
void print(double val){
  std::cout<<val<<std::endl;
}

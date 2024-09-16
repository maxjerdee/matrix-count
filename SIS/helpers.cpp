/* Implementations of helper functions */

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "globals.h"
#include "helpers.h"

// Logarithm of the binomial coefficient
double log_binom(double a, double b){
    return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1);
}

// Logarithm of the factorial
double log_factorial(int n){
    return std::lgamma(n + 1);
}

// Logarithm of the double factorial
double log_factorial2(int n){
    // Check that n is an even integer
    if(n % 2 != 0){
        std::cout << "Error: log_factorial2 called with odd integer.\n";
        exit(0);
    }
    return n/2*std::log(2) + std::lgamma(n/2 + 1);
}

// Overflow protected version of log(e^a + e^b)
double log_sum_exp(double a, double b){
    double a_b_max = std::max(a,b);
    return a_b_max + std::log(std::exp(a - a_b_max) + std::exp(b - a_b_max));
}

// Overflow protected version of log(\sum exp(x_i))
double log_sum_exp(std::vector<double> xs){
    double xs_max = *std::max_element(xs.begin(),xs.end());
    double result = 0;
    for(double x : xs){
        result += std::exp(x - xs_max);
    }
    return xs_max + std::log(result);
}

// Sample from a set of outcomes given log weights (not necessarily normalized)
int sample_log_weights(std::vector<double> log_weights){
    // Normalize the weights
    double log_sum = log_sum_exp(log_weights);
    std::vector<double> weights;
    for(double log_weight : log_weights){
        weights.push_back(std::exp(log_weight - log_sum));
    }

    // Sample according to these weights
    double p = std::rand() / (RAND_MAX + 1.0);
    double culm_weight = 0;
    for(int i = 0; i < weights.size(); i++){
        culm_weight += weights[i];
        if(p <= culm_weight){
            return i;
        }
    }
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
void print(std::string val){
  std::cout<<val<<std::endl;
}
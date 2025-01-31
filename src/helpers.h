/* Helper function declarations */

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <vector>

// Logarithm of the binomial coefficient
double log_binom(double a, double b);

// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);
double log_sum_exp(std::vector<double> xs);

// Sample from a set of outcomes given log weights (not necessarily normalized),
// also return the entropy of that choice
std::pair<int, double> sample_log_weights(std::vector<double> log_weights);

// Sample from a Bernoulli distribution with given probability
std::pair<int, double> sample_bernoulli(double p);

// Logarithm of the factorial
double log_factorial(int n);

// Logarithm of the double factorial
double log_factorial2(int n);

// Return the indices perm which label the row sums from largest to smallest
template <typename T> std::vector<int> argsort(const std::vector<T> &array);

// Erdos-Gallai condition for a degree sequence (whether a simple graph with
// this degree sequence exists)
bool erdos_gallai_condition(const std::vector<int> &ks);

// Printing functions (useful for debugging)
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);
void print(std::string val);

#endif // HELPERS_H

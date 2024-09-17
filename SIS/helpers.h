/* Helper function declarations */

#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <string>

double log_binom(double a, double b);
// Overflow protected version of log(e^a + e^b) for precomputing log_q_vals
double log_sum_exp(double a, double b);
double log_sum_exp(std::vector<double> xs);

std::pair<int,double> sample_log_weights(std::vector<double> log_weights);

double log_factorial(int n);
double log_factorial2(int n);

// Printing functions (useful for debugging)
void print(std::vector<std::vector<int>> table);
void print(std::vector<std::vector<double>> table);
void print(std::vector<int> vec);
void print(std::vector<double> vec);
void print(int val);
void print(double val);
void print(std::string val);

#endif // HELPERS_H
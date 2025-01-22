/* Global variable declarations */

#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>

inline int max_iterations = 1000000; // maximum number of iterations performed (-T flag)
inline int max_time = 60*60; // in seconds (-t flag)

// Global information about the problem
inline int n;
inline int m;
inline double alpha;

// Constants
inline double ALPHA_EPSILON = 1e-7; // Small constant to add to the alpha parameter to avoid poles

// Large arrays for storing the values in the dynamic programming algorithm
inline double ***log_g_i_sP_sT;

inline std::random_device rd;
inline std::mt19937 rng(rd()); // Initialize Mersenne Twister (global RNG)
inline std::uniform_real_distribution<double> dis(0.0, 1.0); // Uniform distribution on [0,1]

#endif // GLOBALS_H
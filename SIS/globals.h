/* Global variable declarations */

#ifndef GLOBALS_H
#define GLOBALS_H

inline int max_iterations = 1000000; // maximum number of iterations performed (-T flag)
inline int max_time = 60; // in seconds (-t flag)

// Large arrays for storing the values in the dynamic programming algorithm
inline double ***log_g_i_sP_sT;

#endif // GLOBALS_H
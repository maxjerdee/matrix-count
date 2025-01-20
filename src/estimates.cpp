// Functions relevant for the moment-matching approximations of the number of matrices

#include <numeric>
#include <vector>
#include <cstdio>

#include "globals.h"
#include "estimates.h"
#include "helpers.h"

// Logarithm of the estimate for the number of symmetric tables with given margin and diagonal sum
double log_Omega_fixed_diagonal(std::vector<int> ks, int m_in){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges
    // Get the alpha value for the estimate
    double alpha_DM;
    if(m_in == 0){
        alpha_DM = alpha_zero_diagonal(n, m);
    }else{
        alpha_DM = alpha_constrained(n, m, m_in);
    }

    // Calculate the log of the estimate
    int m_out = m - m_in;
    double log_Omega = log_binom(m_out + alpha*n*(n-1)/2 - 1,alpha*n*(n-1)/2 - 1) + log_binom(m_in + alpha*n - 1,alpha*n - 1) - log_binom(2*m + n*alpha_DM - 1, n*alpha_DM - 1);
    for(int i = 0; i < n; i++){
        log_Omega += log_binom(ks[i] + alpha_DM - 1, alpha_DM - 1);
    }

    return log_Omega;
}

// Logarithm of the estimate for the number of symmetric tables with given margin and unconstrained diagonal sum
double log_Omega_unconstrained_diagonal(std::vector<int> ks){
    // Extract relevant metadata
    int n = ks.size(); // Number of rows/columns
    int m = std::accumulate(ks.begin(), ks.end(), 0)/2; // Number of edges

    // Get the alpha value for the estimate
    double alpha_DM = alpha_unconstrained(n, m);

    // Calculate the log of the estimate
    double log_Omega = log_binom(m + alpha*n*(n+1)/2 - 1,alpha*n*(n+1)/2 - 1) - log_binom(2*m + n*alpha_DM - 1, n*alpha_DM - 1);
    for(int i = 0; i < n; i++){
        log_Omega += log_binom(ks[i] + alpha_DM - 1, alpha_DM - 1);
    }
    return log_Omega;
}

// alpha moment-matching parameter for an unconstrained diagonal sum
double alpha_unconstrained(int n, int m) {
    double matrix_total = 2 * m;

    double numerator = matrix_total + (n + 1) * (matrix_total + n * (matrix_total - 1) - 2) * alpha;

    double denominator = 2 * (matrix_total - 1) + n * ((n + 1) * alpha + matrix_total - 2);

    double result = numerator / (denominator + ALPHA_EPSILON);
    return result;
}

// alpha moment-matching parameter for a fixed diagonal sum
double alpha_constrained(int n, int m, int diagonal_sum) {
    double matrix_total = 2 * m;
    double mu = (m - diagonal_sum) / double(m);

    double numerator = -(
        4 * m * m * mu * mu * (-1 + n) * (2 + (-1 + n) * n * alpha) +
        4 * m * mu * (-1 + n) * n * alpha * (2 + (-1 + n) * n * alpha) -
        4 * m * m * (-1 + n) * (1 + n * alpha) * (2 + (-1 + n) * n * alpha) +
        (2 * m - 2 * m * mu) * (-2 + n) * (1 + n * alpha) * (matrix_total - diagonal_sum + (-1 + n) * n * alpha)
    );

    double denominator = n * (
        4 * m * m * mu * mu * (-1 + n) * (2 + (-1 + n) * n * alpha) +
        4 * m * mu * (-1 + n) * n * alpha * (2 + (-1 + n) * n * alpha) -
        2 * m * (-1 + n) * (1 + n * alpha) * (2 + (-1 + n) * n * alpha) +
        (2 * m - 2 * m * mu) * (-2 + n) * (1 + n * alpha) * (matrix_total - diagonal_sum + (-1 + n) * n * alpha)
    );

    double result = numerator / (denominator + ALPHA_EPSILON);
    return result;
}

// alpha moment-matching parameter for a zero diagonal sum (case of fixed, but the expression simplifies)
double alpha_zero_diagonal(int n, int m){
    double matrix_total = 2 * m;
    double numerator = -((2 - 3 * n + n * n) * alpha) + matrix_total * (1 + (-1 + n) * (-1 + n) * alpha);

    double denominator = 2 - 2 * matrix_total - n * n * alpha + n * (-2 + matrix_total + alpha);

    double result = numerator / (denominator + ALPHA_EPSILON);
    return result;
}
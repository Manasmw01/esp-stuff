#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "A_array.h"
#include "H_array.h"
#include "initial_state_array.h"
#include "measurements_array.h"
#include "P_array.h"
#include "prediction_array.h"
#include "Q_array.h"
#include "real_array.h"
#include "W_array.h"

#define STATE_SIZE 6  // Number of states n
#define MEAS_SIZE 164  // Number of measurements m
#define SAMPLES 10  // Number of measurements m
#define ERROR_THRESHOLD 0.02
int tot_errors = 0;
void matrix_multiply(double* A, double* B, double* C, int n, int m, int p);
void matrix_add(double* A, double* B, double* C, int n, int m);
void matrix_transpose(double* A, double* AT, int n, int m);
void matrix_inverse(double* A, double* A_inv, int n);
void gauss_inverse(double* A, double* A_inv, int n);
void matrix_subtract(double* A, double* B, double* C, int n);
void print_matrix(double* matrix, int rows, int cols);
void print_vector(double* vector, int size);

    double x[STATE_SIZE];
    double prediction_x[STATE_SIZE];
    double temp_var[STATE_SIZE];
    double temp_meas[MEAS_SIZE];
    double P[STATE_SIZE * STATE_SIZE];
    double A_transpose[STATE_SIZE * STATE_SIZE];
    double Pp[STATE_SIZE * STATE_SIZE];
    double H_transpose[MEAS_SIZE * STATE_SIZE];
    double K[STATE_SIZE * MEAS_SIZE];
    double HP[MEAS_SIZE * STATE_SIZE];
    double HPHT[MEAS_SIZE * MEAS_SIZE];
    double HPHT_R[MEAS_SIZE * MEAS_SIZE];
    double KtH[STATE_SIZE * MEAS_SIZE];
    double KtHP[STATE_SIZE * STATE_SIZE];
    double I[STATE_SIZE * STATE_SIZE];
    double temp1[STATE_SIZE * STATE_SIZE];
    double xp[STATE_SIZE];

void kalman_filter(double* initial, double* P_flat, double* A, double* Q, double* R, double* H, 
                   double* measurements, double* real_out, double* prediction, int iter,     double* xp) {


    // Load initial state and covariance matrix
    for (int i = 0; i < STATE_SIZE; i++) {
        // x[i] = initial[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            // P[i * STATE_SIZE + j] = P_flat[(STATE_SIZE*STATE_SIZE*iter) + i * STATE_SIZE + j];
            // P[i * STATE_SIZE + j] = P_flat[i * STATE_SIZE + j];
            P[i * STATE_SIZE + j] = P_flat[i * STATE_SIZE + j];
            I[i * STATE_SIZE + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    printf("\n");


    // printf("\n\nP2(%d)\n", iter);
    // print_matrix(P, STATE_SIZE, STATE_SIZE);

    // printf("\n\nx2(%d)\n", iter);
    // print_vector(x, STATE_SIZE); // 

    matrix_transpose(A, A_transpose, STATE_SIZE, STATE_SIZE); // A^T

    // printf("A\n");
    // print_matrix(A, STATE_SIZE, STATE_SIZE);

    // printf("A_transpose\n");
    // print_matrix(A_transpose, STATE_SIZE, STATE_SIZE);

    // xp = A * x
    // double xp[STATE_SIZE];
    matrix_multiply(A, x, xp, STATE_SIZE,   STATE_SIZE, 1); // xp = A*x2

    // printf("intermediate state vector\n");
    // print_vector(xp, STATE_SIZE);

    // Pp = A * P * A_transpose
    double Pp_tmp[STATE_SIZE * STATE_SIZE];
    matrix_multiply(A, P, Pp_tmp, STATE_SIZE, STATE_SIZE, STATE_SIZE); // Pp = A * P2

    double Pp_tmp2[STATE_SIZE * STATE_SIZE];
    matrix_multiply(Pp_tmp, A_transpose, Pp_tmp2, STATE_SIZE, STATE_SIZE, STATE_SIZE); // Pp = (A * P2) * A^T

    // Add process noise covariance Q
    matrix_add(Pp_tmp2, W, Pp, STATE_SIZE, STATE_SIZE); // Pp = (A * P2) * A^T + Q
    // printf("\n\nPp final\n");
    // print_matrix(Pp, STATE_SIZE, STATE_SIZE);

    // Compute H_transpose
    matrix_transpose(H, H_transpose, MEAS_SIZE, STATE_SIZE); // H^T

    // printf("H\n");
    // print_matrix(H, MEAS_SIZE, STATE_SIZE);
    // printf("H_transpose\n");
    // print_matrix(H_transpose, STATE_SIZE, MEAS_SIZE);

    // Compute HP
    matrix_multiply(H, Pp, HP, MEAS_SIZE, STATE_SIZE, STATE_SIZE); // H * Pp

    // Print intermediate HP matrix
    // printf("HP\n");
    // print_matrix(HP, MEAS_SIZE, STATE_SIZE);

    // Compute HP * H_transpose
    matrix_multiply(HP, H_transpose, HPHT, MEAS_SIZE, STATE_SIZE, MEAS_SIZE); // (H * Pp) * H^T

    // Print intermediate HPHT matrix
    // printf("HPHT\n");
    // print_matrix(HPHT, MEAS_SIZE, MEAS_SIZE);

    // Compute HPHT + R
    matrix_add(HPHT, Q, HPHT_R, MEAS_SIZE, MEAS_SIZE);  // (H * Pp) * H^T + R 
    // printf("HPHT\n");
    // print_matrix(HPHT, MEAS_SIZE, MEAS_SIZE);

    // Print intermediate HPHT + R matrix
    // printf("HPHT + R\n");
    // print_matrix(HPHT, MEAS_SIZE, MEAS_SIZE);

    // Compute (HPHT + R)^-1
    double HPHT_inv[MEAS_SIZE * MEAS_SIZE];
    // matrix_inverse(HPHT, HPHT_inv, MEAS_SIZE);
    gauss_inverse(HPHT_R, HPHT_inv, MEAS_SIZE); // inv((H * Pp) * H^T + R)

    // printf("inv matrix\n");
    // print_matrix(HPHT_inv, MEAS_SIZE, MEAS_SIZE);


    // Compute K = Pp * H_transpose * (HPHT + R)^-1
    double K_tmp[STATE_SIZE * MEAS_SIZE];
    matrix_multiply(Pp, H_transpose, K_tmp, STATE_SIZE, STATE_SIZE, MEAS_SIZE); // Pp * H^T
    
    matrix_multiply(K_tmp, HPHT_inv, K, STATE_SIZE, MEAS_SIZE, MEAS_SIZE); // K = Pp * H^T * inv((H * Pp) * H^T + R)

    // printf("Kalman Gain\n");
    // print_matrix(K, STATE_SIZE, MEAS_SIZE);

    // Compute z3 - H * xp
    double z3_minus_H_xp[MEAS_SIZE];
    double H_xp[MEAS_SIZE];
    matrix_multiply(H, xp, H_xp, MEAS_SIZE, STATE_SIZE, 1); // H * xp
    for (int i = 0; i < MEAS_SIZE; i++) {
        z3_minus_H_xp[i] = measurements[MEAS_SIZE*(iter) + i] - H_xp[i]; // z3 - H * xp
    }

    for (int i = 0; i < MEAS_SIZE; i++) 
    {
        temp_meas[i] = measurements[MEAS_SIZE*(iter) + i];
        // if(i == 0)
        //     printf("START: %d", MEAS_SIZE*(iter) + i);
        // if(i == MEAS_SIZE - 1)
        //     printf("END: %d", MEAS_SIZE*(iter) + i);
    }
    // printf("Measurements\n");
    // print_vector(temp_meas, MEAS_SIZE);

    // Print intermediate measurement residual
    // print_vector(z3_minus_H_xp, MEAS_SIZE);

    // Compute x3 = xp + K * (z3 - H * xp)
    matrix_multiply(K, z3_minus_H_xp, prediction_x, STATE_SIZE, MEAS_SIZE, 1); // K * (z3 - H * xp)
    for (int i = 0; i < STATE_SIZE; i++) {
        prediction_x[i] = xp[i] + prediction_x[i]; // x3 = xp + K * (z3 - H * xp)
        x[i] = prediction_x[i];
    }

    // Print updated state vector
    // printf("\n\nupdated state vector(2nd measurement in the prediction_array.h file): ");

    // printf("\n\nx3(%d)\n", iter);
    // print_vector(prediction, STATE_SIZE); // 

    // Compute P3 = Pp - K * H * Pp
    matrix_multiply(K, H, KtH, STATE_SIZE, MEAS_SIZE, STATE_SIZE);  // K * H
    matrix_multiply(KtH, Pp, KtHP, STATE_SIZE, STATE_SIZE, STATE_SIZE); // K * H * Pp
    // printf("\n\nPp\n");
    // print_matrix(Pp, STATE_SIZE, STATE_SIZE);
    // printf("KtHP\n");
    // print_matrix(KtHP, STATE_SIZE, STATE_SIZE);
    matrix_subtract(Pp, KtHP, temp1, STATE_SIZE*STATE_SIZE); // P3 = Pp - (K * H * Pp)

    // printf("P3\n");
    // print_matrix(temp1, STATE_SIZE, STATE_SIZE);

    // Update the covariance matrix
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            P_flat[i * STATE_SIZE + j] = temp1[i * STATE_SIZE + j];
        }
    }
}



// Utility function implementations
void gauss_inverse(double* A, double* A_inv, int n) {
    // Augmenting the matrix A with identity matrix of same dimensions
    double augmented[n * 2 * n];

    // Create the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * 2 * n + j] = A[i * n + j];  // A portion
            augmented[i * 2 * n + (j + n)] = (i == j) ? 1.0 : 0.0;  // Identity portion
        }
    }

    // Applying Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find the pivot row
        int pivot_row = i;
        for (int j = i + 1; j < n; j++) {
            if (augmented[j * 2 * n + i] > augmented[pivot_row * 2 * n + i]) {
                pivot_row = j;
            }
        }

        // Swap rows i and pivot_row
        if (pivot_row != i) {
            for (int k = 0; k < 2 * n; k++) {
                double temp = augmented[i * 2 * n + k];
                augmented[i * 2 * n + k] = augmented[pivot_row * 2 * n + k];
                augmented[pivot_row * 2 * n + k] = temp;
            }
        }

        // Make the diagonal elements 1
        double pivot = augmented[i * 2 * n + i];
        for (int k = 0; k < 2 * n; k++) {
            augmented[i * 2 * n + k] /= pivot;
        }

        // Make other elements in the column 0
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = augmented[j * 2 * n + i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented[j * 2 * n + k] -= factor * augmented[i * 2 * n + k];
                }
            }
        }
    }

    // Copy the inverse from the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_inv[i * n + j] = augmented[i * 2 * n + (j + n)];
        }
    }
}

/*
void matrix_inverse(double* A, double* A_inv, int n) {
    int i, j, k;
    
    // Create an augmented matrix with A and the identity matrix
    double* aug = (double*)malloc(n * 2 * n * sizeof(double));
    
    // Initialize augmented matrix: [A | I]
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i * 2 * n + j] = A[i * n + j];      // Copy A into the left half
            aug[i * 2 * n + (n + j)] = (i == j) ? 1 : 0;  // Identity matrix on the right half
        }
    }

    // Perform Gaussian elimination to transform A into I and I into A_inv
    for (i = 0; i < n; i++) {
        // Pivoting: Find the maximum element in the column and swap rows
        double maxEl = fabs(aug[i * 2 * n + i]);
        int maxRow = i;
        for (k = i + 1; k < n; k++) {
            if (fabs(aug[k * 2 * n + i]) > maxEl) {
                maxEl = fabs(aug[k * 2 * n + i]);
                maxRow = k;
            }
        }

        // Swap the current row with the maximum row
        if (maxRow != i) {
            for (k = 0; k < 2 * n; k++) {
                double tmp = aug[i * 2 * n + k];
                aug[i * 2 * n + k] = aug[maxRow * 2 * n + k];
                aug[maxRow * 2 * n + k] = tmp;
            }
        }

        // Make the diagonal element 1, and apply row operations
        double diagEl = aug[i * 2 * n + i];
        for (k = 0; k < 2 * n; k++) {
            aug[i * 2 * n + k] /= diagEl;
        }

        // Make all elements below the pivot zero
        for (j = 0; j < n; j++) {
            if (i != j) {
                double factor = aug[j * 2 * n + i];
                for (k = 0; k < 2 * n; k++) {
                    aug[j * 2 * n + k] -= factor * aug[i * 2 * n + k];
                }
            }
        }
    }

    // Copy the inverse matrix from the augmented matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A_inv[i * n + j] = aug[i * 2 * n + (n + j)];
        }
    }

    free(aug);
}
*/

void matrix_multiply(double* A, double* B, double* C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < m; k++) {
                C[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

void matrix_add(double* A, double* B, double* C, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = A[i * m + j] + B[i * m + j];
        }
    }
}

void matrix_transpose(double* A, double* AT, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT[j * n + i] = A[i * m + j];
        }
    }
}

void matrix_subtract(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Functions to print matrices and vectors
void print_matrix(double* matrix, int rows, int cols) {
    printf("Matrix (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("(Row %d)\t:", i);
        for (int j = 0; j < cols; j++) {
              if (isnan(matrix[i * cols + j])) 
              {
                printf("\nNaN detected during row operations at row %d, col %d\n", i, j);
              }
              else
              {
                printf("%.20f ", matrix[i * cols + j]);
              }
        }
        printf("\n");
    }
}

void print_vector(double* vector, int size) {
    printf("Vector (%d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%.20f ", vector[i]);
    }
    printf("\n");
}

int main() {
    double R[MEAS_SIZE * MEAS_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        x[i] = initial[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            P_flat[i * STATE_SIZE + j] = 0;
            // I[i * STATE_SIZE + j] = (i == j) ? 1.0 : 0.0;
        }
    }
  float sum_sqr_vec = 0.0;
  float abs_diff = 0.0;
    for (int iter = 1; iter < SAMPLES; iter++) 
    {
        float diff_vec[STATE_SIZE];
        float sqr_diff;

        kalman_filter(initial, P_flat, A, Q, R, H, measurements, real_out, prediction, iter, xp);
        for (int i = 0; i < STATE_SIZE; i++) 
        {
            temp_var[i] = prediction[STATE_SIZE*(iter) + i];
            // if(i == 0)
            //     printf("START: %d", STATE_SIZE*(iter) + i);
            // if(i == STATE_SIZE - 1)
            //     printf("END: %d", STATE_SIZE*(iter) + i);
        }
        // printf("Expected\n");
        // print_vector(temp_var, STATE_SIZE);
        printf("final intermediate state vector\n");
        print_vector(xp, STATE_SIZE);
        
        for (int i = 0; i < STATE_SIZE; i++)
            diff_vec[i] = xp[i] - temp_var[i];
        for (int i = 0; i < STATE_SIZE; i++) 
            abs_diff += fabs(diff_vec[i]);

        sqr_diff = abs_diff * abs_diff;
        sum_sqr_vec += sqr_diff;

    }
  sum_sqr_vec = sum_sqr_vec/((STATE_SIZE-1)*STATE_SIZE);
    printf("MSE: %f\n", sum_sqr_vec);

    return 0;
}
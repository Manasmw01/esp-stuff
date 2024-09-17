#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_row_major.h"

#define STATE_SIZE 6  // Number of states n
#define MEAS_SIZE 46  // Number of measurements m

// double HPHT[MEAS_SIZE * MEAS_SIZE];
double HPHT_inv[MEAS_SIZE * MEAS_SIZE];

void gauss_inverse(double* A, double* A_inv, int n);


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
                printf("%f ", matrix[i * cols + j]);
              }
        }
        printf("\n");
    }
}
int main() {
    // double R[MEAS_SIZE * MEAS_SIZE];
    // kalman_filter(initial, P_flat, A, Q, R, H, measurements, real_out, prediction);
    gauss_inverse(matrixx, HPHT_inv, MEAS_SIZE); // inv((H * Pp) * H^T + R)
    printf("inv matrix\n");
    print_matrix(HPHT_inv, MEAS_SIZE, MEAS_SIZE);

    return 0;
}

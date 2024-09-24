#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "A_array_hc.h"

#define STATE_SIZE 4  // Number of states n
#define MEAS_SIZE 4  // Number of measurements m
#define SAMPLES 10  // Number of measurements m
#define ERROR_THRESHOLD 0.02
int tot_errors = 0;
void matrix_multiply(float* A, float* B, float* C, int n, int m, int p);
void matrix_add(float* A, float* B, float* C, int n, int m);
void matrix_transpose(float* A, float* AT, int n, int m);
void matrix_inverse(float* A, float* A_inv, int n);
void gauss_inverse(float* A, float* A_inv, int n);
void matrix_subtract(float* A, float* B, float* C, int n);
void print_matrix(float* matrix, int rows, int cols);
void print_vector(float* vector, int size);
int check_error(float measurements[], float temp_var[], int size);

float x[STATE_SIZE];
float prediction_x[STATE_SIZE];
float temp_var[STATE_SIZE];
float P[STATE_SIZE * STATE_SIZE];
float A_transpose[STATE_SIZE * STATE_SIZE];
float Pp[STATE_SIZE * STATE_SIZE];
float H_transpose[MEAS_SIZE * STATE_SIZE];
float K[STATE_SIZE * MEAS_SIZE];
float HP[MEAS_SIZE * STATE_SIZE];
float HPHT[MEAS_SIZE * MEAS_SIZE];
float KtH[STATE_SIZE * MEAS_SIZE];
float KtHP[STATE_SIZE * STATE_SIZE];
float I[STATE_SIZE * STATE_SIZE];
float temp1[STATE_SIZE * STATE_SIZE];

// Utility function implementations
void gauss_inverse(float* A, float* A_inv, int n) {
    // Augmenting the matrix A with identity matrix of same dimensions
    float augmented[n * 2 * n];

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
                float temp = augmented[i * 2 * n + k];
                augmented[i * 2 * n + k] = augmented[pivot_row * 2 * n + k];
                augmented[pivot_row * 2 * n + k] = temp;
            }
        }

        // Make the diagonal elements 1
        float pivot = augmented[i * 2 * n + i];
        for (int k = 0; k < 2 * n; k++) {
            augmented[i * 2 * n + k] /= pivot;
        }

        // Make other elements in the column 0
        for (int j = 0; j < n; j++) {
            if (j != i) {
                float factor = augmented[j * 2 * n + i];
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



void matrix_multiply(float* A, float* B, float* C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < m; k++) {
                C[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

void matrix_add(float* A, float* B, float* C, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = A[i * m + j] + B[i * m + j];
        }
    }
}

void matrix_transpose(float* A, float* AT, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT[j * n + i] = A[i * m + j];
        }
    }
}

void matrix_subtract(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Functions to print matrices and vectors
void print_matrix(float* matrix, int rows, int cols) {
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

void print_vector(float* vector, int size) {
    printf("Vector (%d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%.20f ", vector[i]);
    }
    printf("\n");
}

int main() 
{
    float R[MEAS_SIZE * MEAS_SIZE];
    printf("A\n");
    // matrix_transpose(A, A_transpose, STATE_SIZE, STATE_SIZE); // A^T
    matrix_multiply(A, A, temp1, STATE_SIZE,   STATE_SIZE, STATE_SIZE); // xp = A*x2
    // gauss_inverse(A, temp1, MEAS_SIZE); // inv((H * Pp) * H^T + R)
    // matrix_subtract(A, A, temp1, STATE_SIZE); // Pp = (A * P2) * A^T + Q
    print_matrix(A, STATE_SIZE, STATE_SIZE);
    print_matrix(temp1, STATE_SIZE, STATE_SIZE);


    // matrix_multiply(H, Pp, HP, MEAS_SIZE, STATE_SIZE, STATE_SIZE); // H * Pp

    return 0;
}
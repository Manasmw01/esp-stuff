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
#define SAMPLES 100  // Number of measurements m
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


    float vec_X[STATE_SIZE];
    float Mat_P[STATE_SIZE * STATE_SIZE];
    float Mat_F[STATE_SIZE * STATE_SIZE];
    float Mat_Q[STATE_SIZE * STATE_SIZE];
    float Mat_R[MEAS_SIZE * MEAS_SIZE];
    float Mat_H[MEAS_SIZE * STATE_SIZE];
    float vec_Z[MEAS_SIZE];


    float Mat_K[STATE_SIZE * MEAS_SIZE];
    float Mat_I[STATE_SIZE*STATE_SIZE];



    float x[STATE_SIZE];
    float prediction_x[STATE_SIZE];
    float prediction_ref[STATE_SIZE];
    float temp_meas[MEAS_SIZE];
    float P[STATE_SIZE * STATE_SIZE];
    float A_transpose[STATE_SIZE * STATE_SIZE];
    float Pp[STATE_SIZE * STATE_SIZE];
    float H_transpose[MEAS_SIZE * STATE_SIZE];
    float K[STATE_SIZE * MEAS_SIZE];
    float HP[MEAS_SIZE * STATE_SIZE];
    float HPHT[MEAS_SIZE * MEAS_SIZE];
    float HPHT_R[MEAS_SIZE * MEAS_SIZE];
    float KtH[STATE_SIZE * MEAS_SIZE];
    float KtHP[STATE_SIZE * STATE_SIZE];
    float I[STATE_SIZE * STATE_SIZE];
    float temp1[STATE_SIZE * STATE_SIZE];
    float xp[STATE_SIZE];
void kalman_filter_new(float* vec_X, float* Mat_P, float* Mat_F, float* Mat_Q, float* Mat_R, float* Mat_H, float* vec_Z) 
{
    float Y[MEAS_SIZE];
    float Mat_S[MEAS_SIZE*MEAS_SIZE];


    float HtF[MEAS_SIZE * STATE_SIZE];
    matrix_multiply(Mat_H, Mat_F, HtF, MEAS_SIZE,   STATE_SIZE, STATE_SIZE); // xp = A*x2

    float H_F_X[MEAS_SIZE];
    matrix_multiply(HtF, vec_X, H_F_X, MEAS_SIZE,   STATE_SIZE, 1); // xp = A*x2

    matrix_subtract(vec_Z, H_F_X, Y, MEAS_SIZE); // P3 = Pp - (K * H * Pp)

    // printf("Y\n");
    // print_vector(Y, MEAS_SIZE);

    float F_Transpose[STATE_SIZE * STATE_SIZE];
    matrix_transpose(Mat_F, F_Transpose, STATE_SIZE, STATE_SIZE); // A^T

    float H_Transpose[STATE_SIZE * MEAS_SIZE];
    matrix_transpose(Mat_H, H_Transpose, MEAS_SIZE, STATE_SIZE); // A^T

    float FtP[STATE_SIZE * STATE_SIZE];
    matrix_multiply(Mat_F, Mat_P, FtP, STATE_SIZE,   STATE_SIZE, STATE_SIZE); 

    float F_P_FT[STATE_SIZE * STATE_SIZE];
    matrix_multiply(FtP, F_Transpose, F_P_FT, STATE_SIZE,   STATE_SIZE, STATE_SIZE); 

    float F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    matrix_add(F_P_FT, Mat_Q, F_P_FT_Q, STATE_SIZE, STATE_SIZE); // Pp = (A * P2) * A^T + Q

    // float Mat_H[MEAS_SIZE * STATE_SIZE];
    // float F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    float H_times_F_P_FT_Q[MEAS_SIZE * STATE_SIZE];
    matrix_multiply(Mat_H, F_P_FT_Q, H_times_F_P_FT_Q, MEAS_SIZE,   STATE_SIZE, STATE_SIZE); 

    // float H_times_F_P_FT_Q[MEAS_SIZE * STATE_SIZE];
    // float H_Transpose[STATE_SIZE * MEAS_SIZE];
    float H_times_F_P_FT_Q_times_HT[MEAS_SIZE * MEAS_SIZE];
    matrix_multiply(H_times_F_P_FT_Q, H_Transpose, H_times_F_P_FT_Q_times_HT, MEAS_SIZE,   STATE_SIZE, MEAS_SIZE); 
    // printf("H_times_F_P_FT_Q_times_HT\n");
    // print_matrix(H_times_F_P_FT_Q_times_HT, MEAS_SIZE, MEAS_SIZE);

    // float H_times_F_P_FT_Q_times_HT[MEAS_SIZE * MEAS_SIZE];
    // float Mat_R[MEAS_SIZE * MEAS_SIZE];
    matrix_add(H_times_F_P_FT_Q_times_HT, Mat_R, Mat_S, MEAS_SIZE, MEAS_SIZE); // Pp = (A * P2) * A^T + Q
    // printf("Mat_S\n");
    // print_matrix(Mat_S, MEAS_SIZE, MEAS_SIZE);

    float S_inv[MEAS_SIZE * MEAS_SIZE];
    gauss_inverse(Mat_S, S_inv, MEAS_SIZE); 

    // float F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    // float H_Transpose[STATE_SIZE * MEAS_SIZE];
    float F_P_FT_Q_times_HT[STATE_SIZE*MEAS_SIZE];
    matrix_multiply(F_P_FT_Q, H_Transpose, F_P_FT_Q_times_HT, STATE_SIZE,   STATE_SIZE, MEAS_SIZE); 

    // float H_Transpose[STATE_SIZE * MEAS_SIZE];
    // float S_inv[MEAS_SIZE * MEAS_SIZE];
    // float Mat_K[STATE_SIZE * MEAS_SIZE];
    matrix_multiply(F_P_FT_Q_times_HT, S_inv, Mat_K, STATE_SIZE,   MEAS_SIZE, MEAS_SIZE); 
    // printf("Mat_K\n");
    // print_matrix(Mat_K, STATE_SIZE, MEAS_SIZE);


    // float Mat_F[STATE_SIZE * STATE_SIZE];
    // float vec_X[STATE_SIZE];
    float FtX[STATE_SIZE];
    matrix_multiply(Mat_F, vec_X, FtX, STATE_SIZE,   STATE_SIZE, 1); 

    // float Mat_K[STATE_SIZE * MEAS_SIZE];
    // float Y[MEAS_SIZE];
    float KtY[STATE_SIZE];
    matrix_multiply(Mat_K, Y, KtY, STATE_SIZE,   MEAS_SIZE, 1); 

    matrix_add(FtX, KtY, vec_X, STATE_SIZE, 1); // Pp = (A * P2) * A^T + Q

    // printf("vec_X\n");
    // print_vector(vec_X, STATE_SIZE);


    // float F_P_FT_Q[STATE_SIZE * STATE_SIZE];

    // float Mat_I[STATE_SIZE*STATE_SIZE];

    // float KtH[STATE_SIZE * STATE_SIZE];
    matrix_multiply(Mat_K, Mat_H, KtH, STATE_SIZE,   MEAS_SIZE, STATE_SIZE); 

    float I_minus_KtH[STATE_SIZE * STATE_SIZE];
    matrix_subtract(Mat_I, KtH, I_minus_KtH, STATE_SIZE*STATE_SIZE); // P3 = Pp - (K * H * Pp)

    // float I_minus_KtH[STATE_SIZE * STATE_SIZE];
    // float F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    // float Mat_P[STATE_SIZE * STATE_SIZE];
    matrix_multiply(I_minus_KtH, F_P_FT_Q, Mat_P, STATE_SIZE,   STATE_SIZE, STATE_SIZE);
    // printf("Mat_P\n");
    // print_matrix(Mat_P, STATE_SIZE, STATE_SIZE);


    
    // mat_P_ = (Mat_I -  Mat_K * mat_H_) * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_);

}
void kalman_filter(float* initial, float* P_flat, float* A, float* Q, float* R, float* H, 
                   float* measurements, float* real_out, float* prediction, int iter,     float* xp) {


    // Load initial state and covariance matrix
    for (int i = 0; i < STATE_SIZE; i++) {
        // x[i] = initial[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            // P[i * STATE_SIZE + j] = P_flat[(STATE_SIZE*STATE_SIZE*iter) + i * STATE_SIZE + j];
            // P[i * STATE_SIZE + j] = P_flat[i * STATE_SIZE + j];
            P[i * STATE_SIZE + j] = P_flat[i * STATE_SIZE + j];
            Mat_I[i * STATE_SIZE + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    // printf("\n");


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
    // float xp[STATE_SIZE];
    matrix_multiply(A, x, xp, STATE_SIZE,   STATE_SIZE, 1); // xp = A*x2

    // printf("intermediate state vector\n");
    // print_vector(xp, STATE_SIZE);

    // Pp = A * P * A_transpose
    float Pp_tmp[STATE_SIZE * STATE_SIZE];
    matrix_multiply(A, P, Pp_tmp, STATE_SIZE, STATE_SIZE, STATE_SIZE); // Pp = A * P2

    float Pp_tmp2[STATE_SIZE * STATE_SIZE];
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
    float HPHT_inv[MEAS_SIZE * MEAS_SIZE];
    // matrix_inverse(HPHT, HPHT_inv, MEAS_SIZE);
    gauss_inverse(HPHT_R, HPHT_inv, MEAS_SIZE); // inv((H * Pp) * H^T + R)

    // printf("inv matrix\n");
    // print_matrix(HPHT_inv, MEAS_SIZE, MEAS_SIZE);


    // Compute K = Pp * H_transpose * (HPHT + R)^-1
    float K_tmp[STATE_SIZE * MEAS_SIZE];
    matrix_multiply(Pp, H_transpose, K_tmp, STATE_SIZE, STATE_SIZE, MEAS_SIZE); // Pp * H^T
    
    matrix_multiply(K_tmp, HPHT_inv, K, STATE_SIZE, MEAS_SIZE, MEAS_SIZE); // K = Pp * H^T * inv((H * Pp) * H^T + R)


    // Compute z3 - H * xp
    float z3_minus_H_xp[MEAS_SIZE];
    float H_xp[MEAS_SIZE];
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
        // printf("(Row %d)\t:", i);
        for (int j = 0; j < cols; j++) {
              if (isnan(matrix[i * cols + j])) 
              {
                printf("\nNaN detected during row operations at row %d, col %d\n", i, j);
              }
              else
              {
                printf("%.15f ", matrix[i * cols + j]);
              }
        }
        printf("\n");
    }
}

void print_vector(float* vector, int size) {
    // printf("Vector (%d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%.15f ", vector[i]);
    }
    printf("\n");
}

int main() {
    float R[MEAS_SIZE * MEAS_SIZE];

    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            Mat_I[i * STATE_SIZE + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < STATE_SIZE; i++) {
        vec_X[i] = initial[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            Mat_P[i * STATE_SIZE + j] = 0;
        }
    }

    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            Mat_F[i * STATE_SIZE + j] = A[i * STATE_SIZE + j];
        }
    }

    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j++) {
            Mat_Q[i * STATE_SIZE + j] = W[i * STATE_SIZE + j];
        }
    }

    for (int i = 0; i < MEAS_SIZE; i++) {
        for (int j = 0; j < MEAS_SIZE; j++) {
            Mat_R[i * MEAS_SIZE + j] = Q[i * MEAS_SIZE + j];
        }
    }

    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < MEAS_SIZE; j++) {
            Mat_H[i * MEAS_SIZE + j] = H[i * MEAS_SIZE + j];
        }
    }

    // printf("vec_X vector:");
    // print_vector(vec_X, STATE_SIZE);

    // printf("\n\nP_flat:\n");
    // print_matrix(Mat_P, STATE_SIZE, STATE_SIZE);

    // printf("\nMat_F:\n");
    // print_matrix(Mat_F, STATE_SIZE, STATE_SIZE);

    // printf("\nMat_Q:\n");
    // print_matrix(Mat_Q, STATE_SIZE, STATE_SIZE);

    // printf("\nMat_R:\n");
    // print_matrix(Mat_R, MEAS_SIZE, MEAS_SIZE);

    // printf("\nMat_H:\n");
    // print_matrix(Mat_H, MEAS_SIZE, STATE_SIZE);

  float sum_sqr_vec = 0.0;
  float abs_diff = 0.0;
    for (int iter = 1; iter < SAMPLES; iter++) 
    {
        float diff_vec[STATE_SIZE];
        float sqr_diff;
        for(int i = MEAS_SIZE*iter; i < MEAS_SIZE*(iter+1); i++)
        {
            vec_Z[i-MEAS_SIZE*iter] = measurements[i];    
        }
        // printf("vec_Z vector:");
        // print_vector(vec_Z, MEAS_SIZE);

        // kalman_filter(initial, P_flat, A, Q, R, H, measurements, real_out, prediction, iter, xp);
        kalman_filter_new(vec_X, Mat_P, Mat_F, Mat_Q, Mat_R, Mat_H, vec_Z);

        printf("\nprediction vector:\t");
        print_vector(vec_X, STATE_SIZE);

        for (int i = 0; i < STATE_SIZE; i++) 
        {
            prediction_ref[i] = prediction[STATE_SIZE*(iter) + i];
        }
        printf("reference vector:\t");
        print_vector(prediction_ref, STATE_SIZE);
        
        for (int i = 0; i < STATE_SIZE; i++)
            diff_vec[i] = vec_X[i] - prediction_ref[i];
        for (int i = 0; i < STATE_SIZE; i++) 
            abs_diff += fabs(diff_vec[i]);

        sqr_diff = abs_diff * abs_diff;
        sum_sqr_vec += sqr_diff;

    }
    sum_sqr_vec = sum_sqr_vec/((STATE_SIZE-1)*STATE_SIZE);
    printf("MSE: %.20f\n", sum_sqr_vec);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef float data_type;

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
void matrix_multiply(data_type* A, data_type* B, data_type* C, int n, int m, int p);
void matrix_add(data_type* A, data_type* B, data_type* C, int n, int m);
void matrix_transpose(data_type* A, data_type* AT, int n, int m);

void LU_inverse(data_type* A, data_type* A_inv, int n);
void gauss_inverse(data_type* A, data_type* A_inv, int n);
void inverse_clean(float new_mat[MEAS_SIZE][MEAS_SIZE], float out[MEAS_SIZE][MEAS_SIZE]);
void matrix_subtract(data_type* A, data_type* B, data_type* C, int n);
void print_matrix(data_type* matrix, int rows, int cols);
void print_vector(data_type* vector, int size);


    data_type vec_X[STATE_SIZE];
    data_type Mat_P[STATE_SIZE * STATE_SIZE];
    data_type Mat_F[STATE_SIZE * STATE_SIZE];
    data_type Mat_Q[STATE_SIZE * STATE_SIZE];
    data_type Mat_R[MEAS_SIZE * MEAS_SIZE];
    data_type Mat_H[MEAS_SIZE * STATE_SIZE];
    data_type vec_Z[MEAS_SIZE];


    data_type Mat_K[STATE_SIZE * MEAS_SIZE];
    data_type Mat_I[STATE_SIZE*STATE_SIZE];



    data_type x[STATE_SIZE];
    data_type prediction_x[STATE_SIZE];
    data_type prediction_ref[STATE_SIZE];
    data_type temp_meas[MEAS_SIZE];
    data_type P[STATE_SIZE * STATE_SIZE];
    data_type A_transpose[STATE_SIZE * STATE_SIZE];
    data_type Pp[STATE_SIZE * STATE_SIZE];
    data_type H_transpose[MEAS_SIZE * STATE_SIZE];
    data_type K[STATE_SIZE * MEAS_SIZE];
    data_type HP[MEAS_SIZE * STATE_SIZE];
    data_type HPHT[MEAS_SIZE * MEAS_SIZE];
    data_type HPHT_R[MEAS_SIZE * MEAS_SIZE];
    data_type KtH[STATE_SIZE * MEAS_SIZE];
    data_type KtHP[STATE_SIZE * STATE_SIZE];
    data_type I[STATE_SIZE * STATE_SIZE];
    data_type temp1[STATE_SIZE * STATE_SIZE];
    data_type xp[STATE_SIZE];
void kalman_filter_new(data_type* vec_X, data_type* Mat_P, data_type* Mat_F, data_type* Mat_Q, data_type* Mat_R, data_type* Mat_H, data_type* vec_Z) 
{
    data_type Y[MEAS_SIZE];
    data_type Mat_S[MEAS_SIZE*MEAS_SIZE];
    data_type Mat_S_2D[MEAS_SIZE][MEAS_SIZE];


    data_type HtF[MEAS_SIZE * STATE_SIZE];
    matrix_multiply(Mat_H, Mat_F, HtF, MEAS_SIZE,   STATE_SIZE, STATE_SIZE); // xp = A*x2

    data_type H_F_X[MEAS_SIZE];
    matrix_multiply(HtF, vec_X, H_F_X, MEAS_SIZE,   STATE_SIZE, 1); // xp = A*x2

    matrix_subtract(vec_Z, H_F_X, Y, MEAS_SIZE); // P3 = Pp - (K * H * Pp)

    data_type F_Transpose[STATE_SIZE * STATE_SIZE];
    matrix_transpose(Mat_F, F_Transpose, STATE_SIZE, STATE_SIZE); // A^T

    data_type H_Transpose[STATE_SIZE * MEAS_SIZE];
    matrix_transpose(Mat_H, H_Transpose, MEAS_SIZE, STATE_SIZE); // A^T

    data_type FtP[STATE_SIZE * STATE_SIZE];
    matrix_multiply(Mat_F, Mat_P, FtP, STATE_SIZE,   STATE_SIZE, STATE_SIZE); 

    data_type F_P_FT[STATE_SIZE * STATE_SIZE];
    matrix_multiply(FtP, F_Transpose, F_P_FT, STATE_SIZE,   STATE_SIZE, STATE_SIZE); 

    data_type F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    matrix_add(F_P_FT, Mat_Q, F_P_FT_Q, STATE_SIZE, STATE_SIZE); // Pp = (A * P2) * A^T + Q

    data_type H_times_F_P_FT_Q[MEAS_SIZE * STATE_SIZE];
    matrix_multiply(Mat_H, F_P_FT_Q, H_times_F_P_FT_Q, MEAS_SIZE,   STATE_SIZE, STATE_SIZE); 

    data_type H_times_F_P_FT_Q_times_HT[MEAS_SIZE * MEAS_SIZE];
    matrix_multiply(H_times_F_P_FT_Q, H_Transpose, H_times_F_P_FT_Q_times_HT, MEAS_SIZE,   STATE_SIZE, MEAS_SIZE); 
    matrix_add(H_times_F_P_FT_Q_times_HT, Mat_R, Mat_S, MEAS_SIZE, MEAS_SIZE); // Pp = (A * P2) * A^T + Q

    data_type S_inv[MEAS_SIZE * MEAS_SIZE];
    data_type S_inv_2D[MEAS_SIZE][MEAS_SIZE]; // Replace data_type with the appropriate type (e.g., float)
    // gauss_inverse(Mat_S, S_inv, MEAS_SIZE); 


// INVERSE CLEAN IMPLEMENTATION STARTS
    for (int i = 0; i < MEAS_SIZE; i++)  
    {
        for (int j = 0; j < MEAS_SIZE; j++) 
        {
            Mat_S_2D[i][j] = Mat_S[i * MEAS_SIZE + j]; // Accessing the 1D array using the row-major order formula
        }
    }
    inverse_clean(Mat_S_2D, S_inv_2D);
    for (int i = 0; i < MEAS_SIZE; i++)  
    {
        for (int j = 0; j < MEAS_SIZE; j++) 
        {
            S_inv[i * MEAS_SIZE + j] = S_inv_2D[i][j];  // Converting 2D element back to 1D
        }
    }
// INVERSE CLEAN IMPLEMENTATION ENDS




    // LU_inverse(Mat_S, S_inv, MEAS_SIZE); 

    // data_type F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    // data_type H_Transpose[STATE_SIZE * MEAS_SIZE];
    data_type F_P_FT_Q_times_HT[STATE_SIZE*MEAS_SIZE];
    matrix_multiply(F_P_FT_Q, H_Transpose, F_P_FT_Q_times_HT, STATE_SIZE,   STATE_SIZE, MEAS_SIZE); 

    // data_type H_Transpose[STATE_SIZE * MEAS_SIZE];
    // data_type S_inv[MEAS_SIZE * MEAS_SIZE];
    // data_type Mat_K[STATE_SIZE * MEAS_SIZE];
    matrix_multiply(F_P_FT_Q_times_HT, S_inv, Mat_K, STATE_SIZE,   MEAS_SIZE, MEAS_SIZE); 
    // printf("Mat_K\n");
    // print_matrix(Mat_K, STATE_SIZE, MEAS_SIZE);


    // data_type Mat_F[STATE_SIZE * STATE_SIZE];
    // data_type vec_X[STATE_SIZE];
    data_type FtX[STATE_SIZE];
    matrix_multiply(Mat_F, vec_X, FtX, STATE_SIZE,   STATE_SIZE, 1); 

    // data_type Mat_K[STATE_SIZE * MEAS_SIZE];
    // data_type Y[MEAS_SIZE];
    data_type KtY[STATE_SIZE];
    matrix_multiply(Mat_K, Y, KtY, STATE_SIZE,   MEAS_SIZE, 1); 

    matrix_add(FtX, KtY, vec_X, STATE_SIZE, 1); // Pp = (A * P2) * A^T + Q

    // printf("vec_X\n");
    // print_vector(vec_X, STATE_SIZE);


    // data_type F_P_FT_Q[STATE_SIZE * STATE_SIZE];

    // data_type Mat_I[STATE_SIZE*STATE_SIZE];

    // data_type KtH[STATE_SIZE * STATE_SIZE];
    matrix_multiply(Mat_K, Mat_H, KtH, STATE_SIZE,   MEAS_SIZE, STATE_SIZE); 

    data_type I_minus_KtH[STATE_SIZE * STATE_SIZE];
    matrix_subtract(Mat_I, KtH, I_minus_KtH, STATE_SIZE*STATE_SIZE); // P3 = Pp - (K * H * Pp)

    // data_type I_minus_KtH[STATE_SIZE * STATE_SIZE];
    // data_type F_P_FT_Q[STATE_SIZE * STATE_SIZE];
    // data_type Mat_P[STATE_SIZE * STATE_SIZE];
    matrix_multiply(I_minus_KtH, F_P_FT_Q, Mat_P, STATE_SIZE,   STATE_SIZE, STATE_SIZE);
    // printf("Mat_P\n");
    // print_matrix(Mat_P, STATE_SIZE, STATE_SIZE);


    
    // mat_P_ = (Mat_I -  Mat_K * mat_H_) * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_);

}

// Function prototypes


// Function to perform LU decomposition of matrix A
void lu_decomposition(data_type* A, int* P, int n) {
    for (int i = 0; i < n; i++) {
        P[i] = i;  // Initialize pivot array
    }

    for (int i = 0; i < n; i++) {
        // Partial pivoting
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k * n + i]) > fabs(A[maxRow * n + i])) {
                maxRow = k;
            }
        }

        // Swap rows in the pivot array
        if (maxRow != i) {
            int temp = P[i];
            P[i] = P[maxRow];
            P[maxRow] = temp;
        }

        // Swap rows in matrix A
        for (int k = 0; k < n; k++) {
            data_type temp = A[i * n + k];
            A[i * n + k] = A[P[i] * n + k];
            A[P[i] * n + k] = temp;
        }

        // LU Decomposition
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] /= A[i * n + i];
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
}

// Function to perform forward and backward substitution to find the inverse
void lu_invert(data_type* A, data_type* A_inv, int* P, int n) {
    // Loop over each column for solving A * X = I
    for (int j = 0; j < n; j++) {
        // Initialize identity matrix column
        data_type identity[n];
        for (int i = 0; i < n; i++) 
        {
            identity[i] = 0;
        }
        identity[j] = 1.0f;

        // Forward substitution (solve L * y = b)
        data_type y[n];
        for (int i = 0; i < n; i++) {
            y[i] = identity[P[i]];
            for (int k = 0; k < i; k++) {
                y[i] -= A[i * n + k] * y[k];
            }
        }

        // Backward substitution (solve U * x = y)
        for (int i = n - 1; i >= 0; i--) {
            A_inv[i * n + j] = y[i];
            for (int k = i + 1; k < n; k++) {
                A_inv[i * n + j] -= A[i * n + k] * A_inv[k * n + j];
            }
            A_inv[i * n + j] /= A[i * n + i];
        }
    }
}

// Utility function to compute the inverse using LU decomposition
void LU_inverse(data_type* A, data_type* A_inv, int n) {
    int P[n];  // Pivot array
    lu_decomposition(A, P, n);  // Perform LU decomposition
    lu_invert(A, A_inv, P, n);  // Invert using the LU factors
}

// Utility function implementations
void gauss_inverse(data_type* A, data_type* A_inv, int n) {
    // Augmenting the matrix A with identity matrix of same dimensions
    data_type augmented[n * 2 * n];

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
                data_type temp = augmented[i * 2 * n + k];
                augmented[i * 2 * n + k] = augmented[pivot_row * 2 * n + k];
                augmented[pivot_row * 2 * n + k] = temp;
            }
        }

        // Make the diagonal elements 1
        data_type pivot = augmented[i * 2 * n + i];
        for (int k = 0; k < 2 * n; k++) {
            augmented[i * 2 * n + k] /= pivot;
        }

        // Make other elements in the column 0
        for (int j = 0; j < n; j++) {
            if (j != i) {
                data_type factor = augmented[j * 2 * n + i];
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

void matrix_multiply(data_type* A, data_type* B, data_type* C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < m; k++) {
                C[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

void matrix_add(data_type* A, data_type* B, data_type* C, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = A[i * m + j] + B[i * m + j];
        }
    }
}

void matrix_transpose(data_type* A, data_type* AT, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT[j * n + i] = A[i * m + j];
        }
    }
}

void matrix_subtract(data_type* A, data_type* B, data_type* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

// Functions to print matrices and vectors
void print_matrix(data_type* matrix, int rows, int cols) {
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

void print_vector(data_type* vector, int size) {
    // printf("Vector (%d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%.15f ", vector[i]);
    }
    printf("\n");
}

int main() {
    data_type R[MEAS_SIZE * MEAS_SIZE];

    for (int i = 0; i < STATE_SIZE; i++) 
    {
        for (int j = 0; j < STATE_SIZE; j++) 
        {
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
            printf("\nMat_F:\n");
            print_matrix(Mat_F, STATE_SIZE, STATE_SIZE);

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

  data_type sum_sqr_vec = 0.0;
  data_type abs_diff = 0.0;
    for (int iter = 1; iter < SAMPLES; iter++) 
    {
        data_type diff_vec[STATE_SIZE];
        data_type sqr_diff;
        for(int i = MEAS_SIZE*iter; i < MEAS_SIZE*(iter+1); i++)
        {
            vec_Z[i-MEAS_SIZE*iter] = measurements[i];    
        }
        // printf("vec_Z vector:");
        // print_vector(vec_Z, MEAS_SIZE);

        // kalman_filter(initial, P_flat, A, Q, R, H, measurements, real_out, prediction, iter, xp);
        kalman_filter_new(vec_X, Mat_P, Mat_F, Mat_Q, Mat_R, Mat_H, vec_Z);


        for (int i = 0; i < STATE_SIZE; i++) 
        {
            prediction_ref[i] = prediction[STATE_SIZE*(iter) + i];
        }
        printf("\nreference(%d):", iter);
        print_vector(prediction_ref, STATE_SIZE);
        
        printf("prediction(%d):", iter);
        print_vector(vec_X, STATE_SIZE);
        for (int i = 0; i < STATE_SIZE; i++)
            diff_vec[i] = vec_X[i] - prediction_ref[i];
        for (int i = 0; i < STATE_SIZE; i++) 
            abs_diff += fabs(diff_vec[i]);

        sqr_diff = abs_diff * abs_diff;
        sum_sqr_vec += sqr_diff;

    }
    sum_sqr_vec = sum_sqr_vec/((STATE_SIZE-1)*STATE_SIZE);
    printf("MSE: %e\n", sum_sqr_vec);
    // std::cout << "MSE is = \n" << sum_sqr_vec << std::endl;  

    return 0;
}


void inverse_clean(float new_mat[MEAS_SIZE][MEAS_SIZE], float out[MEAS_SIZE][MEAS_SIZE])
{

		 float ratio;
		 int i,j,k;

		 if(MEAS_SIZE == 2){
			 float a = new_mat[0][0];
			 float b = new_mat[0][1];
			 float c = new_mat[1][0];
			 float d = new_mat[1][1];

			 float det = (a * d) - (b * c);

			 if (det == 0) {
			     printf("The matrix is not invertible.\n");
			     return;
			 }

			 out[0][0] = d / det;
			 out[0][1] = (-1) * b / det;
			 out[1][0] = (-1) * c / det;
			 out[1][1] = a / det;

			 return;
		 }

		 /* Applying Gauss Jordan Elimination */
		 for(i = 0; i < MEAS_SIZE; i++)
		 {
			  for(j = 0; j < MEAS_SIZE; j++)
			  {
				   if(i != j)
				   {
					    ratio = new_mat[j][i]/new_mat[i][i];
					    for(k = 0; k < MEAS_SIZE; k++)
					    {

					    	if(i == MEAS_SIZE-1){
					    		if(k == 0){//Calc the diagonal element first
					    			new_mat[j][j] = new_mat[j][j] - ratio*new_mat[i][j];
					    			//out[j][j] = (out[j][j] - ratio*out[i][j]) / new_mat[j][j];
					    		}
					    		else if(k == j){
					    			new_mat[j][0] = new_mat[j][0] - ratio*new_mat[i][0];
					    			//out[j][0] = (out[j][0] - ratio*out[i][0]) / new_mat[j][j];
					    		}
					    		else{
					    			new_mat[j][k] = new_mat[j][k] - ratio*new_mat[i][k];
					    			//out[j][k] = (out[j][k] - ratio*out[i][k]) / new_mat[j][j];
					    		}

					    		out[j][k] = (out[j][k] - ratio*out[i][k]) / new_mat[j][j];
					    		//out[j][j] = out[j][j] / new_mat[j][j];
					    	}
					    	else{

								new_mat[j][k] = new_mat[j][k] - ratio*new_mat[i][k];

								if(i > 0)
									out[j][k] = out[j][k] - ratio*out[i][k];
								else{ //(i == 0)
									if(i == k)
										out[i][k] = 1;
									else
										out[i][k] = 0;
									if(j == k){
										if(i == k)
											out[j][k] = 1 - ratio;
										else
											out[j][k] = 1;
										//out[j][k] = 1 - ratio*out[i][k];
									}
									else{
										if(i == k)
											out[j][k] = 0 - ratio;
										else
											out[j][k] = 0;
										//out[j][k] = 0 - ratio*out[i][k];
									}
								}
					    	}

					    }

				   }
//				   else{//(i == j)
//					   out[i][j] = out[i][j] / new_mat[i][i];
//				   }
			  }
		 }

		 /* Row Operation to Make Principal Diagonal to 1 */
		 for(i = 0; i < MEAS_SIZE; i++)
		 {
			 out[MEAS_SIZE-1][i] = out[MEAS_SIZE-1][i] / new_mat[MEAS_SIZE-1][MEAS_SIZE-1];
		 }

//		 /* Row Operation to Make Principal Diagonal to 1 */
//		 for(i = 0; i < N; i++)
//		 {
//			  for(j = 0; j < N;j++)
//			  {
//			   	out[i][j] = out[i][j]/new_mat[i][i];
//			  }
//		 }

		 return;
}

#include <stdio.h>
#include<math.h>
#define N 4 // Size of the square matrices
typedef float FPDATA;

    FPDATA dt = 0.01;

// Function prototypes
void getCofactor(FPDATA A[N][N], FPDATA temp[N][N], int p, int q, int n);
FPDATA determinant(FPDATA A[N][N], int n);
void adjoint(FPDATA A[N][N], FPDATA adj[N][N]);
void inverse(FPDATA A[N][N], FPDATA inverse[N][N]);
void multiplyMatrices(FPDATA A[N][N], FPDATA B[N][N], FPDATA result[N][N]);
void printMatrix(FPDATA matrix[N][N]);

// Function to get the cofactor of A[p][q] in temp[][]. n is current dimension of A[][]
void getCofactor(FPDATA A[N][N], FPDATA temp[N][N], int p, int q, int n) {
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // Copying into temporary matrix only those element which are not in given row and column
            if (row != p && col != q) {
                temp[i][j++] = A[row][col];

                // Row is filled, so increase row index and reset column index
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

// Function to calculate the determinant of matrix A[][]
FPDATA determinant(FPDATA A[N][N], int n) {
    FPDATA det = 0; // Initialize result

    // Base case: if matrix contains single element
    if (n == 1)
        return A[0][0];

    FPDATA temp[N][N]; // To store cofactors

    // Sign multiplier
    int sign = 1;

    // Iterate for each element of first row
    for (int f = 0; f < n; f++) {
        // Getting cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        // Recursive call to get determinant of temp[][] matrix
        det += sign * A[0][f] * determinant(temp, n - 1);

        // Terms are to be added with alternate sign
        sign = -sign;
    }

    return det;
}

// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(FPDATA A[N][N], FPDATA adj[N][N]) {
    if (N == 1) {
        adj[0][0] = 1;
        return;
    }

    // Temporarily store cofactors
    FPDATA temp[N][N];
    int sign = 1;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, N);

            // Sign of adj[j][i] positive if sum of row and column indexes is even
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the transpose of the cofactor matrix
            adj[j][i] = (sign) * (determinant(temp, N - 1));
        }
    }
}

// Function to calculate the inverse of the matrix A[N][N] and store it in inverse[N][N].
void inverse(FPDATA A[N][N], FPDATA inverse[N][N]) {
    // Find determinant of A[][]
    FPDATA det = determinant(A, N);

    // If determinant is 0, then inverse doesn't exist
    if (det == 0) {
        printf("Inverse of matrix is not possible\n");
        return;
    }

    // Find adjoint
    FPDATA adj[N][N];
    adjoint(A, adj);

    // Find inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i][j] = adj[i][j] / det;
}

// Function to multiply two matrices A and B and store the result in result
void multiplyMatrices(FPDATA A[N][N], FPDATA B[N][N], FPDATA result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void cholesky(FPDATA A[N][N], FPDATA L[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < (i+1); j++) {
            FPDATA sum = 0;
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = sqrtf(A[j][j] - sum);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
}

void forward_substitution(FPDATA L[N][N], FPDATA b[N], FPDATA y[N]) {
    for (int i = 0; i < N; i++) {
        FPDATA sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
}

void backward_substitution(FPDATA L[N][N], FPDATA y[N], FPDATA x[N]) {
    for (int i = N - 1; i >= 0; i--) {
        FPDATA sum = 0;
        for (int j = i + 1; j < N; j++) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }
}

void multiplyMatrixVector(FPDATA A[N][N], FPDATA vector[N], FPDATA result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = 0;
        for (int j = 0; j < N; j++) {
            result[i] += A[i][j] * vector[j];
        }
    }
}

void transposeMatrix(FPDATA matrix[N][N], FPDATA result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[j][i] = matrix[i][j];
        }
    }
}

void addMatrices(FPDATA A[N][N], FPDATA B[N][N], FPDATA result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtractMatrices(FPDATA A[N][N], FPDATA B[N][N], FPDATA result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
}

void copymat(FPDATA A[N][N], FPDATA result[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = A[i][j];
        }
    }
}

void subtractVectors(FPDATA vector1[N], FPDATA vector2[N], FPDATA result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = vector1[i] - vector2[i];
    }
}

void addVectors(FPDATA vector1[N], FPDATA vector2[N], FPDATA result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

// Function to print the matrix
void printMatrix(FPDATA matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%f ", matrix[i][j]);
        printf("\n");
    }
}

void printVector(FPDATA vec[N]) {
    for (int i = 0; i < N; i++) {
            printf("%f \t", vec[i]);
    }
    printf("\n");
}

// Driver program
int main() {


    FPDATA inverseA[N][N]; // To store inverse of A
    FPDATA result[N][N]; // To store the result of matrix multiplication

    FPDATA Xp[N]; 


    FPDATA X_acc[] = {   0.013375, 0.07029, 0.050267, 0.040236, 0.147412,
                        0.352218, 0.335913, 0.153616, -0.108037, -0.06378};

    FPDATA Y_acc[] = {   0.090298, 0.149292, 0.180564, 0.200428, 0.165192,
                        0.193588, 0.15041, 0.187971, 0.10673, 0.134393};

    FPDATA X_GPS[] = {   4028186.036, 4028186.036, 4028186.036, 4028186.036, 4028186.036,
                        4028186.036, 4028186.036, 4028186.036, 4028186.036, 4028186.036};

    FPDATA Y_GPS[] = {   -4433.544201, -4433.544201, -4433.544201, -4433.544201, -4433.544201, 
                        -4433.544201, -4433.544201, -4433.544201, -4433.544201, -4433.544201};

    FPDATA vel_x[10];    
    FPDATA X_pos[10];    
    FPDATA vel_y[10];    
    FPDATA Y_pos[10];   

    FPDATA Zk[N];
    FPDATA L[N][N] = {0}; // Lower triangular matrix


    // FPDATA X0[N];
    FPDATA X[N];
    FPDATA K[N][N];


    FPDATA tmp_mat1[N][N];
    FPDATA tmp_mat2[N][N];
    FPDATA tmp_mat3[N][N];
    FPDATA tmp_trans[N][N];
    FPDATA tmp_vec[N];
    FPDATA tmp_vec2[N];

    vel_x[0] = X_acc[0]*dt;
    X_pos[0] = X_GPS[0] + vel_x[0]*dt;

    // inital values for velocity and position Y
    vel_y[0] = Y_acc[0]*dt;
    Y_pos[0] = Y_GPS[0] + vel_y[0]*dt;


    for (int i = 1; i < 10; i++)
    {
        vel_x[i] = vel_x[i-1] + X_acc[i-1]*dt;
        X_pos[i] = X_pos[i-1] + vel_x[i]*dt;
        
        vel_y[i] = vel_y[i-1] + Y_acc[i-1]*dt;
        Y_pos[i] = Y_pos[i-1] + vel_y[i]*dt;
    }

    FPDATA phi[N][N] = {
        {1, dt, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, dt},
        {0, 0, 0, 1}
    };

    FPDATA Q[N][N] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
  
    FPDATA H[N][N] = {
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0}
    };

    FPDATA R[N][N] = {
        {25, 0, 0, 0},
        {0, 400, 0, 0},
        {0, 0, 25, 0},
        {0, 0, 0, 400}
    };  

    FPDATA Pp[N][N] = {
        {200, 0, 0, 0},
        {0, 200, 0, 0},
        {0, 0, 200, 0},
        {0, 0, 0, 200}
    };  

        X[0] = X_GPS[0];
        X[1] = 0;
        X[2] = Y_GPS[0];
        X[3] = 0;
    for (int i = 1; i < 10; i++)
    {
        Zk[0] = X_GPS[i];
        Zk[1] = X_pos[i];
        Zk[2] = Y_GPS[i];
        Zk[3] = Y_pos[i];

        
        multiplyMatrixVector(phi, X, Xp); // Xp = phi * X0;
        
        // Pp = phi * Pp * phi' + Q
        multiplyMatrices(phi, Pp, tmp_mat1); 
        transposeMatrix(phi, tmp_trans);
        multiplyMatrices(tmp_mat1, tmp_trans, tmp_mat2);
        addMatrices(tmp_mat2, Q, Pp);
        
        // End Pp = phi * Pp * phi' + Q
        // printf("Pp :\n");
        // printMatrix(Pp);


        // Compute Kalman Gain
        // K = Pp * H' * inv(H * Pp * H' + R);
        multiplyMatrices(H, Pp, tmp_mat1); 
        transposeMatrix(H, tmp_trans);
        multiplyMatrices(tmp_mat1, tmp_trans, tmp_mat2);
        addMatrices(tmp_mat2, R, tmp_mat1);
        inverse(tmp_mat1, tmp_mat3);
        
        
        cholesky(tmp_mat1, L);
        FPDATA identity[N][N] = {0}; // Identity matrix
        FPDATA temp[N]; // Temporary array
        for (int i = 0; i < N; i++) {
            identity[i][i] = 1;
            forward_substitution(L, identity[i], temp);
            backward_substitution(L, temp, identity[i]);
        }
        if(i == 5)
        {
            printf("Gauss Jordan Elimination Inverse :\n");
            printMatrix(tmp_mat3);

            printf("Cholesky Inverse :\n");
            printMatrix(identity);
        }
        
       
        multiplyMatrices(Pp, tmp_trans, tmp_mat1); 
        multiplyMatrices(tmp_mat1, tmp_mat3, K); 
        // END COMPUTE KALMAN GAIN

        
        //Update
        // X(:,i+1) = Xp + K * (Zk - H * Xp);
        
        multiplyMatrixVector(H, Xp, tmp_vec); // H * Xp
        subtractVectors(Zk, tmp_vec, tmp_vec2);

        multiplyMatrixVector(K, tmp_vec2, tmp_vec); // K * (Zk - H * Xp)
        addVectors(Xp, tmp_vec, X);


        //Update
        // Pp = Pp - K*H*Pp;
        multiplyMatrices(K, H, tmp_mat1); 
        multiplyMatrices(tmp_mat1, Pp, tmp_mat2);  // K*H*Pp
        subtractMatrices(Pp, tmp_mat2, tmp_mat1);
        copymat(tmp_mat1, Pp);
        // Update Pp done
    }
        printf("Pp :\n");
        printMatrix(Pp);

        printf("X :\n");
        printVector(X);

        printf("K :\n");
        printMatrix(K);

    return 0;
}

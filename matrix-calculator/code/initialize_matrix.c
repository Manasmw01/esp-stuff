#include "matrix.h"
#include "matrix_functions.h"

void initializeMatrix(Matrix *matrix, int size){ // Called after declaring a matrix
    setSize(matrix, size); // matrix is already a pointer, so no need for &matrix
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            matrix->values[i][j] = 0; // Initializes all values to 0
        }
    }
}
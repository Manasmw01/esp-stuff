#include "matrix.h"

int setSize(Matrix *matrix, int size){ // Requires a pointer to the matrix
    if((size == 2) || (size == 3) || (size == 4)){ // Valid sizes
        matrix->size = size;
        return(1);
    } else{
        return(-1);
    }
}

int setValue(Matrix *matrix, int i, int j, double value){ // Requires a pointer to the matrix
    int matrix_size = getSize(*matrix);
    if(((0 <= i) && (i < matrix_size)) && ((0 <= j) && (j < matrix_size))){ // Valid position
        matrix->values[i][j] = value;
        return(1);
    } else{
        return(-1);
    }
}

int getSize(Matrix matrix){
    return(matrix.size);
}

double getValue(Matrix matrix, int i, int j){
    int matrix_size = getSize(matrix);
    if(((0 <= i) && (i < matrix_size)) && ((0 <= j) && (j < matrix_size))){ // Valid position
        return(matrix.values[i][j]);
    }
}
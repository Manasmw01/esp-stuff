#include <stdio.h>
#include "matrix.h"

void printMatrix(Matrix matrix){
    int matrix_size = getSize(matrix);
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            printf("%6.2lf\t", getValue(matrix, i, j)); // Prints the matrix's value, rounded to 2 decimal places, with 6 preceding spots, and a tab separating values
        }
        printf("\n");
    }
    printf("\n");
}
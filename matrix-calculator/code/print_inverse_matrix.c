#include <stdio.h>
#include "matrix.h"

void printInverseMatrix(Matrix A, Matrix I){
    int matrix_size = getSize(A);
    for(int i = 0; i < matrix_size; i++){ // Rows
        for(int j = 0; j < matrix_size; j++){ // A columns
            printf("%6.2lf\t", getValue(A, i, j)); // Prints the matrix's value, rounded to 2 decimal places, with 6 preceding spots, and a tab separating values
        }
        printf("\t|\t"); // Separating line
        for(int j = 0; j < matrix_size; j++){ // I columns
            printf("%6.2lf\t", getValue(I, i, j)); // Prints the matrix's value, rounded to 2 decimal places, with 6 preceding spots, and a tab separating values
        }
        printf("\n");
    }
    printf("\n");
}
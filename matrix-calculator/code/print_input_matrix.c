#include <stdio.h>
#include "matrix.h"

void printInputMatrix(Matrix matrix, int next_position){
    int position_counter = 0; // To determine where the 'next_position' really is in the matrix
    int matrix_size = getSize(matrix);
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            if(position_counter < next_position){
                printf("%6.2lf\t", getValue(matrix, i, j)); // Prints the matrix's value, rounded to 2 decimal places, with 6 preceding spots, and a tab separating values
            } else if(position_counter == next_position){ // Indicating where the user must place a value
                printf("%6c\t", 'X'); // Prints an X with 6 preceding spots, and a tab separating values
            } else{
                printf("%6c\t", '-'); // Prints a - with 6 preceding spots, and a tab separating values
            }
            position_counter++;
        }
        printf("\n");
    }
    printf("\n");
}
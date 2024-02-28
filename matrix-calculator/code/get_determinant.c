#include "matrix.h"

double getDeterminant(Matrix matrix){
    double determinant = 0.0;
    Matrix temp_matrix;
    int row_index = 0, column_index = 0;

    switch(getSize(matrix)){
        case 2: // Determinant of a 2x2 matrix = ad - bc
            determinant = (getValue(matrix, 0, 0) * getValue(matrix, 1, 1)) - (getValue(matrix, 0, 1) * getValue(matrix, 1, 0));
            break;

        case 3: // Recursively calls getDeterminant on 3x3 matrices; cofactor method along first column
            initializeMatrix(&temp_matrix, 2);

            for(int m = 0; m < 3; m++){ // Rows
                row_index = 0, column_index = 0;
                for(int i = 0; i < 3; i++){ // Rows, barring when i == m
                    if(i != m){
                        for(int j = 1; j < 3; j++){ // Columns
                            setValue(&temp_matrix, row_index, column_index, getValue(matrix, i, j));
                            if(column_index < 1){ // [0][0] -> [0][1]
                                column_index++;
                            } else{
                                column_index = 0; // [1][0] -> [1][1]
                                row_index++;
                            }
                        }
                    }
                }

                if(m % 2 == 0){ // Even => positive
                    determinant += (getValue(matrix, m, 0) * 1 * getDeterminant(temp_matrix));
                } else{ // Odd => negative
                    determinant += (getValue(matrix, m, 0) * -1 * getDeterminant(temp_matrix));
                }
            }
            break;

        case 4: // Recursively calls getDeterminant on 4x4 matrices; cofactor method along first column
            initializeMatrix(&temp_matrix, 3);

            for(int m = 0; m < 4; m++){ // Rows
                row_index = 0, column_index = 0;
                for(int i = 0; i < 4; i++){ // Rows, barring when i == m
                    if(i != m){
                        for(int j = 1; j < 4; j++){ // Columns
                            setValue(&temp_matrix, row_index, column_index, getValue(matrix, i, j));
                            if(column_index < 2){ // [0][0] -> [0][1] -> [0][2] -> [1][1] -> [1][2] -> [2][1] -> [2][2]
                                column_index++;
                            } else{
                                column_index = 0; // [1][0] -> [2][0]
                                row_index++;
                            }
                        }
                    }
                }

                if(m % 2 == 0){ // Even => positive
                    determinant += (getValue(matrix, m, 0) * 1 * getDeterminant(temp_matrix));
                } else{ // Odd => negative
                    determinant += (getValue(matrix, m, 0) * -1 * getDeterminant(temp_matrix));
                }
            }
            break;

        default: // Not a 2x2, 3x3, or 4x4 matrix
            determinant = -1;
    }
    return determinant;
}
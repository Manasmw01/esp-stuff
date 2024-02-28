#include <stdio.h>
#include "matrix.h"
#include "row_operations.h"

Matrix getRREF(Matrix matrix){
    Matrix temp = getREF(matrix);
    int matrix_size = getSize(temp);

    double curr_value;
    double factor;
    double leading_value;
    for(int i = 0; i < matrix_size; i++){ // Setting all leading entries to 1
        for(int j = 0; j < matrix_size; j++){
            curr_value = getValue(temp, i, j);
            if((curr_value <= -0.005) || (0.005 <= curr_value)){ // If curr_value doesn't round to zero
                factor = (1 / curr_value); // (factor)(curr_value) = 1
                rowMultiply(&temp, i, factor);
                printf("R%d(%.2lf)\n", i, factor);
                break;
            }
        }
    }
    printMatrix(temp);
    for(int i = matrix_size - 1; i >= 1; i--){ // Every column except the first, starting from the last
        for(int j = 0; j < matrix_size; j++){
            leading_value = getValue(temp, i, j);
            if(leading_value != 0){ // Need to make all entries above it in the column equal to 0
                for(int k = 0; k < j; k++){
                    curr_value = getValue(temp, k, j);
                    if(curr_value != 0){ // Needs to be turned into zero via row operations
                        factor = ((curr_value * -1) / leading_value); // curr_value + (factor)(leading_value) = 0
                        rowMultiplyAndAdd(&temp, k, j, factor);
                        if(factor < 0){ // Negative
                            printf("R%d - (%.2lf)R%d\n", (k + 1), (factor * -1), (j + 1));
                        } else{ // Positive
                            printf("R%d + (%.2lf)R%d\n", (k + 1), factor, (j + 1));
                        }
                    }
                }
                printMatrix(temp);
                break;
            }
        }
    }

    printf("\nReduced Row Echelon Form:\n");
    printMatrix(temp);

    return(temp);
}
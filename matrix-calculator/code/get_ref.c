#include <stdio.h>
#include "matrix.h"
#include "row_operations.h"

Matrix getREF(Matrix matrix){
    Matrix temp;
    copyMatrix(matrix, &temp);
    int matrix_size = getSize(temp);
    printMatrix(temp);

    
    int all_zero;
    double leading_value;
    double curr_value;
    double factor;
    for(int i = 0; i < matrix_size - 1; i++){
        all_zero = 1; // 0 if all entries in the current column are zero, 1 otherwise
        if(getValue(temp, i, i) == 0){
            all_zero = 0;
            for(int j = i + 1; j < matrix_size; j++){ // i = column & j = row
                if(getValue(temp, j, i) != 0){ // Find first non-zero entry in the colum
                    rowSwap(&temp, j, i);
                    printf("R%d <-> R%d\n", (i + 1), (j + 1));
                    printMatrix(temp);
                    all_zero = 1;
                    break;
                }
            }
        }
        if(all_zero == 0){ // All entries in the current column are already zero
            continue;
        }
        leading_value = getValue(temp, i, i);
        for(int j = i + 1; j < matrix_size; j++){
            curr_value = getValue(temp, j, i);
            if(curr_value != 0){ // Needs to be turned into zero via row operations
                factor = ((curr_value * -1) / leading_value); // curr_value + (factor)(leading_value) = 0
                rowMultiplyAndAdd(&temp, j, i, factor);
                if(factor < 0){ // Negative
                    printf("R%d - (%.2lf)R%d\n", (j + 1), (factor * -1), (i + 1));
                } else{ // Positive
                    printf("R%d + (%.2lf)R%d\n", (j + 1), factor, (i + 1));
                }
            }
        }
        printMatrix(temp);
    }

    printf("\nRow Echelon Form:\n");
    printMatrix(temp);

    return(temp);
}
#include "matrix.h"

#ifndef ROW_OPERATIONS_H
#define ROW_OPERATIONS_H

void rowSwap(Matrix *matrix, int row1, int row2); // Swaps two rows in a matrix by swapping the corresponding values (to avoid directly accessing the matrix's rows)

void rowMultiply(Matrix *matrix, int row, double factor); // Multiplies all the values in a matrix's row by a factor

void rowMultiplyAndAdd(Matrix *matrix, int addRow, int multiplyRow, double factor); // Adds the values of one row, all multiplied by a factor, to the corresponding values in another row

#endif
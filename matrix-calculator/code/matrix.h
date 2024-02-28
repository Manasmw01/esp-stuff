#ifndef MATRIX_H
#define MATRIX_H

#define MAX_SIZE 4 // This program accepts square matrices of dimensions 2x2, 3x3, or 4x4

typedef struct{
    int size;
    double values[MAX_SIZE][MAX_SIZE]; // Values are doubles (rounded to 2 decimal places when printed)
} Matrix;

void getMatrix(Matrix *matrix); // Gets the user's input and initializes a matrix

void initializeMatrix(Matrix *matrix, int size); // Called after declaring a Matrix

Matrix copyMatrix(Matrix originalMatrix, Matrix *newMatrix); // Copies an old matrix into a new one

int setSize(Matrix *matrix, int size);

int setValue(Matrix *matrix, int i, int j, double value);

int getSize(Matrix matrix);

double getValue(Matrix matrix, int i, int j);

void printMatrix(Matrix matrix); // Prints a matrix

void printInputMatrix(Matrix matrix, int next_position); // Prints an input matrix (for use in getMatrix)

void printInverseMatrix(Matrix A, Matrix I); // Prints a matrix in the form (A|I) (for use in getInverse)

double getDeterminant(Matrix matrix); // Calculates and returns the Determinant of a matrix

Matrix getREF(Matrix matrix); // Calculates and returns the Row Echelon Form of a matrix

Matrix getRREF(Matrix matrix); // Calculates and returns the Reduced Row Echelon Form of a matrix

Matrix getInverse(Matrix matrix); // Calculates and returns the Inverse of a matrix

#endif
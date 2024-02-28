#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(void){
    Matrix user_matrix;
    int user_input = -1;
    int play_flag = 1;

    printf("Welcome, this is a matrix calculator.\nIt accepts square matrices of dimensions 2x2, 3x3, or 4x4.\nIt can calculate the matrix's determinant, row echelon form, reduced row echelon form, and inverse.\n\n");
    printf("Please input the size and values of your matrix:\n\n");
    getMatrix(&user_matrix);

    while(play_flag == 1){
        printf("\n");
        printf("----------------------------------------------------------\n");
        printMatrix(user_matrix);
        printf("Options\n\t[1]: Print the matrix\n\t[2]: Reset the matrix\n\t[3]: Calculate the Determinant of the matrix\n\t[4]: Calculate the Row Echelon Form of the matrix\n\t[5]: Calculate the Reduced Row Echelon Form of the matrix\n\t[6]: Calculate the Inverse of the matrix\n\t[7]: Exit the program\nPlease make a choice (enter the corresponding number): ");
        while((scanf("%d", &user_input) != 1) || (user_input != 1 && user_input != 2 && user_input != 3 && user_input != 4 && user_input != 5 && user_input != 6 && user_input != 7)){
            printf("Invalid input. Please make a choice by choosing a valid number (1, 2, 3, 4, 5, 6, or 7): ");
            while(getchar() != '\n'){
                continue;
            }
        }

        switch(user_input){
            case 1:
                break;
            case 2:
                printf("\n");
                getMatrix(&user_matrix);
                break;
            case 3:
                printf("\nDeterminant of the matrix: %.2lf\n", getDeterminant(user_matrix));
                break;
            case 4:
                printf("\nREF =>\n");
                getREF(user_matrix);
                break;
            case 5:
                printf("\nRREF =>\n");
                getRREF(user_matrix);
                break;
            case 6:
                printf("\nInverse =>\n");
                getInverse(user_matrix);
                break;
            case 7:
                play_flag = 0;
                break;
        }
        printf("\n");
    }

    printf("Thank you for using the matrix calculator.\n");
    printf("~End of Program~\n");
    return 0;
}
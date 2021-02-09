#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#define BUFFER 64
#define save_params false
#define load_params true

typedef struct matrix {
    unsigned int shape[2];
    double **array;
} matrix;

void print_matrix(matrix *mat);
matrix* create_matrix_rand(const int h, const int w);
matrix* create_matrix_zeros(const int h, const int w);
matrix* create_weights(const int h, const int w);
matrix* create_zeros_ones(const int h, const int w);
matrix* extract_from_file(char* filename, int row_count);
void cpy_matrix(matrix* result, matrix* a);
void matmul(matrix* result, matrix *x, matrix *y);
void element_wise(matrix* result, matrix* x, matrix* y, char operation);
void transpose(matrix* result, matrix* a);
void scale(matrix* result, matrix *a, double factor);
void save_matrix(matrix* mat, char* file_name);
void free_matrix(matrix* a);

#endif 
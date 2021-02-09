#ifndef __UTILS_H__
#define __UTILS_H__

#include "matrix.h"

#define filename_chars 100

float score(matrix* predictions, matrix* ground_truths, bool print);
void calculate_dAL(matrix* dAL, matrix* AL, matrix* Y);
void sig_der(matrix* result, matrix* Z);
void relu_der(matrix* result, matrix* Z);
void sigmoid(matrix* result, matrix *Z);
void relu(matrix* result, matrix* Z);
char ** get_filenames(char* file, int num_names);


#endif 
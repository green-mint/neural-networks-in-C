#ifndef __LAYER_H__
#define __LAYER_H__

#include "utils.h"
#include <stdbool.h>

typedef struct layer{
    int n;
    char activation;
    matrix *weights;
    matrix *bias;
    matrix *A_prev;
    matrix *Z;
    matrix *A;
    matrix *dA;
    matrix *product_W_A_prev;
    matrix *act_der;
    matrix *dZ;
    matrix *dW;
    matrix *scaled_dW;
    matrix *db;
    matrix *scaled_db;
    matrix *trans_W;
    matrix *trans_A_prev;
    matrix *prod_dZ_trans_A_prev;
} layer;

void relu_der(matrix* result, matrix* Z);
void sigmoid(matrix* result, matrix *Z);
void relu(matrix* result, matrix* Z);
void initialize_layer(layer *l, const int neurons, const int prev_neurons, const char activation, const int examples, bool load_from_file);
void update_parameters_layer(layer *l, double learning_rate);
matrix* propogate_forward_layer(layer *l, matrix *A_prev);
matrix* propogate_backward_layer(layer *l, matrix* A_prev, matrix* dA, matrix* dA_prev);
void free_layer(layer *l);
void sig_der(matrix* result, matrix* Z);

#endif 
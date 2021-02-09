#ifndef __MODEL_H__
#define __MODEL_H__

#include "utils.h"
#include "layer.h"
#include <stdbool.h>

typedef struct model {
    unsigned int num_layers;
    layer** layers;
} model;

void initialize_model(model *m, int num_layers, int model_neurons[], int num_features, int examples, bool load_from_file);
void propogate_forward_model(model *m, matrix* X);
double calculate_cost(matrix* AL, matrix* y);
void propogate_back_model(model *m, matrix* AL, matrix* y, matrix* X);
void calculate_dAL(matrix* dZL, matrix* AL, matrix* Y);
void train_model(model* m, matrix* X , matrix* y, double learning_rate, int epochs, matrix* predictions);
void save_model(model* m, char* dir_name);
void predict_model(model* m, matrix* X, matrix* predictions);
void free_model(model* m);
#endif
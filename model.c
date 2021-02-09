#include "model.h"

void initialize_model(model *m, int num_layers, int model_neurons[], int num_features, int examples, bool load_from_file) {

    m->num_layers = num_layers;    
    m->layers = malloc(num_layers * sizeof(layer*));

    m->layers[0] = malloc(sizeof(layer));
    initialize_layer(m->layers[0],model_neurons[0],num_features,'r',examples, load_from_file);

    for (int i=1; i<m->num_layers; i++) {
        m->layers[i] = malloc(sizeof(layer));
        initialize_layer(m->layers[i],model_neurons[i],model_neurons[i-1],'r',examples, load_from_file);
    }

    m->layers[num_layers-1] = malloc(sizeof(layer));
    initialize_layer(m->layers[num_layers-1],model_neurons[num_layers-1],model_neurons[num_layers-2],'s',examples, load_from_file);
}

void propogate_forward_model(model *m, matrix* X) {

    propogate_forward_layer(m->layers[0],X);
    for (int i=1; i<m->num_layers; i++) {
        propogate_forward_layer(m->layers[i],m->layers[i-1]->A);
    }
}

double calculate_cost(matrix* AL, matrix* y) {
    int m = y->shape[1];
    double cost = 0;
    for (int i=0; i<m; i++) {
        if (AL->array[0][i] == 0) AL->array[0][i] += 0.000000000000001;
        else if (AL->array[0][i] == 1) AL->array[0][i] -= 0.000000000000001;
        cost += y->array[0][i] * log(AL->array[0][i]) + (1 - y->array[0][i]) * log(1 - AL->array[0][i]);
    }
    cost /= -m;
    return cost;
}

 
void propogate_back_model(model *m, matrix* AL, matrix* y, matrix* X) {

    calculate_dAL(m->layers[m->num_layers - 1]->dA,AL,y);
    propogate_backward_layer(m->layers[m->num_layers - 1],m->layers[m->num_layers - 2]->A,m->layers[m->num_layers - 1]->dA,m->layers[m->num_layers - 2]->dA);

    for (int i = m->num_layers - 2; i > 0; i--) {
        propogate_backward_layer(m->layers[i],m->layers[i-1]->A,m->layers[i]->dA,m->layers[i-1]->dA);
    }

    propogate_backward_layer(m->layers[0],X,m->layers[0]->dA,NULL);
}

void update_parameters_model(model* m ,double learning_rate) {
    for (int i=0; i<m->num_layers; i++) {
        update_parameters_layer(m->layers[i],learning_rate);
    }
}



void train_model(model* m, matrix* X , matrix* y, double learning_rate, int epochs, matrix* predictions) {
    double cost = 0;
    for (int i = 0; i < epochs; i++)
    {
        propogate_forward_model(m,X);
        propogate_back_model(m,m->layers[m->num_layers - 1]->A,y,X);
        update_parameters_model(m,learning_rate);
        if ((i+1) % 50 == 0) {
            printf("Loss after %03d iterations: %lf  ",i+1,calculate_cost(m->layers[m->num_layers - 1]->A,y));
            predict_model(m,X,predictions);
            score(predictions,y,false);
        }
    }
    
}

void predict_model(model* m, matrix* X, matrix* predictions) {
    //assert(predictions->shape[1] == X->shape[1]);
    propogate_forward_model(m,X);
    cpy_matrix(predictions,m->layers[m->num_layers - 1]->A);

    for (int i=0; i<predictions->shape[1]; i++) {
        predictions->array[0][i] = (predictions->array[0][i] > 0.5) ? 1 : 0;
    }
}

void save_model(model* m, char* dir_name) {
    char file_path[50];
    for (int i=0; i<m->num_layers; i++) {

        sprintf(file_path,"%sweights%d",dir_name,m->layers[i]->weights->shape[0]);
        printf("%s\n",file_path);
        save_matrix(m->layers[i]->weights,file_path);

        sprintf(file_path,"%sbias%d",dir_name,m->layers[i]->bias->shape[0]);
        printf("%s\n",file_path);
        save_matrix(m->layers[i]->bias,file_path);
    }
}

void free_model(model* m) {
    for (int i=0; i<m->num_layers; i++) {
        free_layer(m->layers[i]);
    }
    free(m);
}
#include "layer.h"

void initialize_layer(layer *l, const int neurons, const int prev_neurons, const char activation, const int examples, bool load_from_file) {
    l->n = neurons;
    l->activation = activation;
    // char filename[10];
    // sprintf(filename,"weights%d",neurons);
    //l->weights = extract_from_file(filename,neurons);
    if (load_from_file) {
        char file_path[50];

        sprintf(file_path,"saved_model/weights%d",neurons);
        l->weights = extract_from_file(file_path,neurons);
    
        sprintf(file_path,"saved_model/bias%d",neurons);
        l->bias = extract_from_file(file_path,neurons);
    }
    else {
        l->weights = create_weights(neurons,prev_neurons);
        l->bias = create_matrix_zeros(neurons,1);
    }
    l->A_prev = create_matrix_zeros(prev_neurons,examples);
    l->Z = create_matrix_zeros(neurons,examples);
    l->A = create_matrix_zeros(neurons,examples);
    l->dA = create_matrix_zeros(neurons,examples);
    l->product_W_A_prev = create_matrix_zeros(neurons,examples);
    l->act_der = create_matrix_zeros(neurons,examples);
    l->dZ = create_matrix_zeros(neurons,examples);
    l->dW = create_matrix_zeros(neurons, prev_neurons);
    l->db = create_matrix_zeros(neurons,1);
    l->trans_W = create_matrix_zeros(prev_neurons,neurons);
    l->trans_A_prev = create_matrix_zeros(examples,prev_neurons);
    l->prod_dZ_trans_A_prev = create_matrix_zeros(neurons,prev_neurons);
    l->scaled_dW = create_matrix_zeros(neurons,prev_neurons);
    l->scaled_db = create_matrix_zeros(neurons,1);
}

matrix* propogate_forward_layer(layer *l, matrix *A_prev) {

    matmul(l->product_W_A_prev,l->weights,A_prev);
    element_wise(l->Z,l->product_W_A_prev,l->bias,'+');
    if (l->activation == 'r') relu(l->A,l->Z);
    else sigmoid(l->A,l->Z);

    return l->A;

}

matrix* propogate_backward_layer(layer *l, matrix*A_prev, matrix* dA, matrix* dA_prev) {
    int m = A_prev->shape[1]; //number of examples (shape of A_prev)

    if (l->activation == 'r') {
        relu_der(l->act_der, l->Z);
        element_wise(l->dZ,dA,l->act_der,'*');
    }
    else {
        cpy_matrix(l->dZ,dA);
    }
    if (dA_prev != NULL) {
        transpose(l->trans_W,l->weights);
        matmul(dA_prev,l->trans_W,l->dZ);    
    }
    
    //dW = scale(element_wise(matmul(dZ,transpose(l->cache[0])),scale(l->weights,lambda),'+'),1.0/m);
    //printf("VALUE: %lf\n",A_prev->array[0][0]);
    transpose(l->trans_A_prev,A_prev);
    matmul(l->prod_dZ_trans_A_prev,l->dZ,l->trans_A_prev);
    
    scale(l->dW,l->prod_dZ_trans_A_prev,1.0/m);

    for (int i=0; i<l->db->shape[0]; i++) {
        l->db->array[i][0] = 0;
        for (int j=0; j<l->dZ->shape[1]; j++) {
            l->db->array[i][0] += l->dZ->array[i][j];
        }
        l->db->array[i][0] /= (float) m;
    }

    return dA_prev;
}

void update_parameters_layer(layer *l, double learning_rate) {
    scale(l->scaled_dW,l->dW,-(learning_rate));
    element_wise(l->weights,l->weights,l->scaled_dW,'+');
    scale(l->scaled_db,l->db,-(learning_rate));
    element_wise(l->bias,l->bias,l->scaled_db,'+');
}





void free_layer(layer *l) {
    free_matrix(l->bias);
    free_matrix(l->weights);
    free_matrix(l->dW);
    free_matrix(l->db);
    free_matrix(l->Z);
    free_matrix(l->product_W_A_prev);
    free_matrix(l->prod_dZ_trans_A_prev);
    free_matrix(l->trans_A_prev);
    free_matrix(l->trans_W);
    free_matrix(l->dZ);
    free_matrix(l->act_der);
    free_matrix(l->scaled_dW);
    free_matrix(l->scaled_db);
    free_matrix(l->A);
    free_matrix(l->dA);
    free(l);
}
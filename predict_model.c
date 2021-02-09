#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include "configs.h"
#include "model.h"

char *test_file = "demo";

int main() {
    int demo_imgs = atoi(getenv("DEMO_IMGS"));
    char ** filenames = get_filenames("fnames",demo_imgs);
    model* neural_network = malloc(sizeof(model));
    matrix* X = extract_from_file(test_file,N_FEATURES);
    matrix* predictions = create_matrix_zeros(1,demo_imgs);
    initialize_model(neural_network,NUM_LAYERS,model_neurons,N_FEATURES,demo_imgs,load_params);
    predict_model(neural_network,X,predictions);
    
    for (int i=0; i<demo_imgs; i++) {
        if (predictions->array[0][i]) {
            printf("\033[0;32m");
            printf("%-70s is ",filenames[i]);
            printf("a cat\n");
        }
        else {
            printf("\033[0;31m");
            printf("%-70s is ",filenames[i]);
            printf("not a cat\n");
        }
    }
}
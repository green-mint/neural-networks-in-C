#include <stdlib.h>
#include <stdbool.h>
#include "model.h"
#include "configs.h"

char *test_file = "test_x";

int main() {
    printf("Testing the model.....");
    model*neural_network = malloc(sizeof(model));
    matrix*X = extract_from_file(test_file,N_FEATURES);
    matrix* y = extract_from_file("test_y",1);
    matrix*predictions = create_matrix_zeros(1,TEST_EXAMPLES);
    initialize_model(neural_network,NUM_LAYERS,model_neurons,N_FEATURES,TEST_EXAMPLES,load_params);
    predict_model(neural_network,X,predictions);
    printf("Peformance Metrics on the test data are as follows:\n");
    score(predictions,y,true);
}
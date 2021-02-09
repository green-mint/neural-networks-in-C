#include "configs.h"
#include "model.h"


int main() {
    //srand(time(NULL));
    model* neural_network = malloc(sizeof(model));
    matrix* X = extract_from_file("X",N_FEATURES);
    matrix* y = extract_from_file("y",1);
    matrix* predictions = create_matrix_zeros(1,TRAIN_EXAMPLES);
    initialize_model(neural_network,NUM_LAYERS,model_neurons,N_FEATURES,TRAIN_EXAMPLES,save_params);

    train_model(neural_network,X,y,LEARNING_RATE,EPOCHS,predictions);
    
    save_model(neural_network,"saved_model/");

}
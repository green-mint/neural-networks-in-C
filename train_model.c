#include "configs.h"
#include "model.h"


int main() {
    printf("\033[1;31m");
    printf("\n....Training the model on %d images and %d iterations....\n\n",TRAIN_EXAMPLES,EPOCHS);
    //srand(time(NULL));
    model* neural_network = malloc(sizeof(model));
    matrix* X = extract_from_file("X",N_FEATURES);
    matrix* y = extract_from_file("y",1);
    matrix* predictions = create_matrix_zeros(1,TRAIN_EXAMPLES);
    initialize_model(neural_network,NUM_LAYERS,model_neurons,N_FEATURES,TRAIN_EXAMPLES,save_params);

    train_model(neural_network,X,y,LEARNING_RATE,EPOCHS,predictions);
    
    printf("\033[1;31m");
    printf("\n...........Saving the model..........\n\n");
    printf("\033[0;37m");
    save_model(neural_network,"saved_model/");

}
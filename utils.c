#include "utils.h"

float score(matrix* predictions, matrix* ground_truths, bool print) {
    float sum=0, fp=0, fn=0;
    for (int i=0; i<predictions->shape[1]; i++) {
        if (predictions->array[0][i] == ground_truths->array[0][i]) sum++;
        else {
            if (predictions->array[0][i] == 1 && ground_truths->array[0][i] == 0) fp++;
            if (predictions->array[0][i] == 0 && ground_truths->array[0][i] == 1) fn++;
        }
    }
    printf("Accuracy: %lf\n",sum/predictions->shape[1]);
    if (print) {
        printf("False Positives: %lf\n",fp/predictions->shape[1]);
        printf("False Negatives: %lf\n",fn/predictions->shape[1]);
    }

    return sum/predictions->shape[1];
}

void calculate_dAL(matrix* dAL, matrix* AL, matrix* Y) {
    for (int i=0; i<Y->shape[1]; i++) {

        dAL->array[0][i] = AL->array[0][i] - Y->array[0][i];
        // double y = Y->array[0][i], al = AL->array[0][i];
        // if (al == 1 || al == 0) {
        //     AL->array[0][i] += 0.00000000000001;
        //     al += 0.00000000000001;
        // }
        // dAL->array[0][i] = -(y/al - (1 - y)/(1 - al));
    }
}

void relu(matrix* result, matrix* Z) {
    assert(result->shape[0] == Z->shape[0] && result->shape[1] == Z->shape[1]); 
    
    for (int i=0; i<result->shape[0]; i++) {
        for (int j=0; j<result->shape[1]; j++) {
            result->array[i][j] = (Z->array[i][j] > 0) ? Z->array[i][j] : 0;
        }
    }
}

void sigmoid(matrix* result, matrix *Z) {

    assert(result->shape[0] == Z->shape[0] && result->shape[1] == Z->shape[1]); 

    for (int i=0; i<result->shape[0]; i++) {
        for (int j=0; j<result->shape[1]; j++) {
            result->array[i][j] = 1.0/ (1 + exp(-(Z->array[i][j])));
        }
    }
}

void relu_der(matrix* result, matrix* Z) {
    
    assert(result->shape[0] == Z->shape[0] && result->shape[1] == Z->shape[1]);

    for (int i=0; i<result->shape[0]; i++) {
        for (int j=0; j<result->shape[1]; j++) {
            result->array[i][j] = (Z->array[i][j] > 0) ? 1 : 0;
        }
    }
}

void sig_der(matrix* result, matrix* Z) {
    assert(result->shape[0] == Z->shape[0] && result->shape[1] == Z->shape[1]);

    for (int i=0; i<result->shape[0]; i++) {
        for (int j=0; j<result->shape[1]; j++) {
            double s = 1.0/ (1 + exp(-(Z->array[i][j])));
            result->array[i][j] = s * (1 - s);   
        }
    }

}

char ** get_filenames(char* file, int num_names) {
    int counter = 0;
    FILE* f = fopen(file,"r");
    char buffer[filename_chars];
    char** filenames = malloc(num_names * sizeof(char *));
    while(fgets(buffer,filename_chars,f) != NULL) {
        filenames[counter] = malloc(filename_chars * sizeof(char));
        buffer[strlen(buffer) - 1] = '\0';
        strcpy(filenames[counter],buffer);
        counter++;
    }
    return filenames;
}
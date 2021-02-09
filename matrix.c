#include "matrix.h"

double _GaussianSample();

void matmul_seq(matrix* result, matrix* x, matrix* y) {
    assert(x->shape[1] == y->shape[0]);
    assert(result->shape[0] == x->shape[0] && result->shape[1] == y->shape[1]);

    for (int i=0; i<x->shape[0]; i++) {
        for (int j=0; j<y->shape[1]; j++) {
            result->array[i][j] = 0;
            
            for (int a=0; a<x->shape[1]; a++) {
                result->array[i][j] += x->array[i][a] * y->array[a][j];
            }
        }
    }
}

void matmul(matrix* result, matrix* A, matrix* B) {
    assert(A->shape[1] == B->shape[0]);
    assert(result->shape[0] == A->shape[0] && result->shape[1] == B->shape[1]);

    #pragma omp parallel shared(A,B,result)
    {
        int i,j,k;

        #pragma omp for schedule(static)
        for(i = 0; i < result->shape[0]; ++i) {
            
            for(j = 0; j < result->shape[1]; ++j) {
                result->array[i][j] = 0;
            }

            for(k = 0; k < A->shape[1]; ++k) {
                
                for(j = 0; j < result->shape[1]; ++j){
                    result->array[i][j] += A->array[i][k] * B->array[k][j];
                }
            }
        }
    }
}

void element_wise(matrix* result, matrix* x, matrix* y, char operation) {

    assert(result->shape[0] == x->shape[0] && result->shape[1] == x->shape[1]);

    char broadcast;

    if (x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1]) broadcast = 'n';
    else if (x->shape[0] == y->shape[0] && y->shape[1] == 1) broadcast = 'c';
    else if (x->shape[1] == y->shape[1] && y->shape[0] == 1) broadcast = 'r';
    else {
        broadcast = 'f';
        printf("ELEMENT WISE ERROR\n");
    }
    
    for (int i=0; i<x->shape[0]; i++) {
        for (int j=0; j<x->shape[1]; j++) {
            switch (broadcast)
            {
                case 'n':
                    switch (operation) {
                        case '+': result->array[i][j] = x->array[i][j] + y->array[i][j]; break;
                        case '*': result->array[i][j] = x->array[i][j] * y->array[i][j]; break;
                        case '-': result->array[i][j] = x->array[i][j] - y->array[i][j]; break;
                    }
                    break;
                
                case 'c':
                    switch (operation) {
                        case '+': result->array[i][j] = x->array[i][j] + y->array[i][0]; break;
                        case '*': result->array[i][j] = x->array[i][j] * y->array[i][0]; break;
                        case '-': result->array[i][j] = x->array[i][j] - y->array[i][0]; break;
                    }
                    break;

                case 'r':
                    switch (operation) {
                        case '+': result->array[i][j] = x->array[i][j] + y->array[0][j]; break;
                        case '*': result->array[i][j] = x->array[i][j] * y->array[0][j]; break;
                        case '-': result->array[i][j] = x->array[i][j] - y->array[0][j]; break;
                    }
                    break;
            }
        }
    }
}


void print_matrix(matrix *mat) {
    printf("[");
    for (int i=0; i<mat->shape[0]; i++) {
        printf("[");
        for (int j=0; j<mat->shape[1]; j++) {
            printf("%lf, ",mat->array[i][j]);
        }
        printf("\b\b]\n");
    }
    printf("SHAPE: (%d, %d)\n",mat->shape[0],mat->shape[1]);
    printf("\n");
}

matrix* create_matrix_rand(const int h, const int w) {
    matrix *result = malloc(sizeof(matrix));
    assert(result != NULL);
    result->shape[0] = h;
    result->shape[1] = w;
    result->array = malloc(h * sizeof(double));
    for (int i=0; i<h; i++) {
        result->array[i] = malloc(w * sizeof(double));
        for (int j=0; j<w; j++) {
            result->array[i][j] = rand() % 10;
        }
    }
    return result;
}

matrix* create_matrix_zeros(const int h, const int w) {
    matrix *result = malloc(sizeof(matrix));
    assert(result != NULL);
    result->shape[0] = h;
    result->shape[1] = w;
    result->array = malloc(h * sizeof(double));
    for (int i=0; i<h; i++) {
        result->array[i] = malloc(w * sizeof(double));
        for (int j=0; j<w; j++) {
            result->array[i][j] = 0;
        }
    }
    return result;
}

matrix* create_weights(const int h, const int w) {
    matrix* weights = malloc(sizeof(matrix));
    assert(weights != NULL);
    weights->shape[0] = h; weights->shape[1] = w;

    weights->array = malloc(h * sizeof(double*));
    assert(weights->array != NULL);
    for (int i=0; i<h; i++) {
        weights->array[i] = malloc(w * sizeof(double));
        assert(weights->array[i] != NULL);

        for (int j=0; j<w; j++) {
            weights->array[i][j] = _GaussianSample() * sqrt(2.0 / w);
        }
    }
    return weights;
}

matrix* create_zeros_ones(const int h, const int w) {
    matrix *result = malloc(sizeof(matrix));
    assert(result != NULL);
    result->shape[0] = h;
    result->shape[1] = w;
    result->array = malloc(h * sizeof(double));
    for (int i=0; i<h; i++) {
        result->array[i] = malloc(w * sizeof(double));
        for (int j=0; j<w; j++) {
            result->array[i][j] = rand() % 2;
        }
    }
    return result;
}

matrix* extract_from_file(char* filename, int row_count) {
    FILE* file = fopen(filename, "r");

    // Will hold the values for both the rows
    double** image = malloc(row_count * sizeof(double*));
    for (int i=0; i<row_count; i++) {
        image[i] = malloc(sizeof(double));
    }
    
    int* column_count = malloc(row_count * sizeof(int));

    for (int row = 0, flag = 1 && row < row_count; flag && row<row_count; row++) {
        // Temporarily stores the floating values
        char content[BUFFER] = {};

        for (int i = 0, curr_column_count = 0;;i++) {
            // Gets a single character
            char c = fgetc(file);
            if (c == ' ' || c == '\n' || c == EOF) {
                // Converting the string to a double
                image[row][curr_column_count] = strtod(content, NULL);
                // Priniting the converted number (optional)
                //printf("%0.64Lf\n", image[row][curr_column_count]);

                // Reallocating the array for the next in comming integer
                image[row] = realloc(image[row], (++curr_column_count + 1) * sizeof(double));
                // Setting the string back to zero
                memset(content, 0, BUFFER);
                // Storing the columns for the current row
                column_count[row] = curr_column_count;
                // Resetting the counter for the next number
                i = 0;

                // This means that the current row has ended
                if (c == '\n') {
                    break;
                }
                // If reached end of file then exits both the loops
                if (c == EOF) {
                    flag = 0;
                    break;
                }
            }

            // Appends the newly read character to the end of the string
            content[i] = c;
        }
        //printf("\n");
    }
    matrix*a = malloc(sizeof(matrix));
    a->array = image;
    a->shape[0] = row_count; a->shape[1] = column_count[0];
    return a;
}


void transpose(matrix* result, matrix* a) {
    assert(result->shape[0] == a->shape[1] && result->shape[1] == a->shape[0]);

    for (int i=0; i<result->shape[0]; i++) {
        for (int j=0; j<result->shape[1]; j++) {
            result->array[i][j] = a->array[j][i];
        }
    }
}

void scale(matrix* result, matrix *a, double factor) {

    assert(result->shape[0] == a->shape[0] && result->shape[1] == a->shape[1]);

    for (int i=0; i<result->shape[0]; i++) {

        for (int j=0; j<result->shape[1]; j++) {
            result->array[i][j] = factor * (a->array[i][j]);
        }
    }
}

void free_matrix(matrix* a) {

    if (a != NULL) {
        //printf("%d\t%d\n",a->shape[0], a->shape[1]);
        for (int i=0; i<a->shape[0]; i++) {
        if (a->array[i] != NULL) free(a->array[i]);
        }
        if (a->array != NULL) free(a->array);
        if (a != NULL) free(a);
    }
}

void cpy_matrix(matrix* result, matrix* a) {
    assert(result->shape[0] == a->shape[0] && result->shape[1] == a->shape[1]);

    for (int i=0; i<a->shape[0]; i++) {
        for (int j=0; j<a->shape[1]; j++) {
            result->array[i][j] = a->array[i][j];
        }
    }
}

void save_matrix(matrix* mat, char* file_name) {
    
    FILE *f = fopen(file_name,"w");

    for (int i=0; i<mat->shape[0]; i++) {
        for (int j=0; j<mat->shape[1]; j++) {
            if (j == (mat->shape[1] - 1)) {
                fprintf(f,"%.20lf",mat->array[i][j]);
                
            }
            else {
                fprintf(f,"%.20lf ",mat->array[i][j]);
            }
        }
        fprintf(f,"\n");
    }
    fclose(f);
}


double _GaussianSample() {
    double v1=( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
    double v2=( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

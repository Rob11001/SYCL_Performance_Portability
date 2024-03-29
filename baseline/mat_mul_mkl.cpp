#include <iostream>
#include "mkl.h"
#include <chrono>

//#define DEBUG

using namespace std;
using namespace chrono;

int main(int argc, char **argv) {
    float *A, *B, *C;
    size_t N, M, K, i, j;
    float alpha = 1.0f, beta = 0.0f;

    if(argc != 4) {
        fprintf(stderr, "Usage: %s <N> <M> <K>\n", argv[0]);

        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);

    A = (float *) mkl_malloc(N * M * sizeof(float), 64);
    B = (float *) mkl_malloc(M * K * sizeof(float), 64);
    C = (float *) mkl_malloc(N * K * sizeof(float), 64);

    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      
      return EXIT_FAILURE;
    }

    // Initialization
    for(int i {0}; i < N * M; i++)
        A[i] = rand() % 5;
    
    for(int i {0}; i < M * K; i++)
        B[i] = rand() % 5;
    
    for(int i {0}; i < N * K; i++)
        C[i] = 0.0f;

    auto start = steady_clock::now();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, K, M, alpha, A, M, B, K, beta, C, K);

    auto end = steady_clock::now();

    #ifdef DEBUG
        cout << "Elapsed time in milliseconds: " << duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
        
        if(N < 32 && M < 32 && K < 32) {
            for(int i {0}; i < N ; i++) {
                for(int j {0}; j < M; j++)
                    std::cout << "A[" << i << "][" << j << "] = " << A[i * M + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
            
            for(int i {0}; i < M ; i++) {
                for(int j {0}; j < K; j++)
                    std::cout << "B[" << i << "][" << j << "] = " << B[i * K + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;

            for(int i {0}; i < N ; i++) {
                for(int j {0}; j < K; j++)
                    std::cout << "C[" << i << "][" << j << "] = " << C[i * K + j] << " ";
                std::cout << std::endl;
            }
        }
        
    #endif

    #ifndef DEBUG
        cout << duration_cast<chrono::milliseconds>(end - start).count() << " ";
    #endif

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
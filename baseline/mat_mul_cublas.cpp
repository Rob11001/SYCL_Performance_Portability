#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

//#define DEBUG

int main (int argc, char ** argv) {
    // For cublas library
    cublasStatus_t stat;
    cublasHandle_t handle;
    size_t N, M, K;

    if(argc != 4) {
        fprintf(stderr, "Usage: %s <N> <M> <K>\n", argv[0]);

        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);

    // Allocate matrix (see if can be use C++ classes)
    // Host data
    float *A_h = static_cast<float *>(malloc(sizeof(float) * N * M));
    float *B_h = static_cast<float *>(malloc(sizeof(float) * M * K));
    float *C_h = static_cast<float *>(malloc(sizeof(float) * N * K));
	
    // Device data
    float *A_d, *B_d, *C_d; // device data
    cudaMalloc((void **) &A_d, N * M * sizeof(float));
    cudaMalloc((void **) &B_d, sizeof(float) * M * K);
    cudaMalloc((void **) &C_d, sizeof(float) * N * K);

    // Data initialization

	// Initialization
    for(int i {0}; i < N * M; i++)
        A_h[i] = 1.0f;
    
    for(int i {0}; i < M * K; i++)
        B_h[i] = 1.0f;
    
    for(int i {0}; i < N * K; i++)
        C_h[i] = 0.0f;
    
    // cuBLAS initialization
    stat = cublasCreate(&handle);              
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    auto start = steady_clock::now();
    // Data movement 
    cudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, M * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(C_d, 0, N * K * sizeof(float));

   
    // Matrix mul
    const float alpha = 1, beta = 0;
    auto kernel_start = steady_clock::now();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, B_d, K, A_d, M, &beta, C_d, K);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Matrix product failed\n");
        cudaFree (A_d);
        cudaFree (B_d);
        cudaFree (C_d);
        cublasDestroy(handle);

        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();
    auto kernel_end = steady_clock::now();
    
    // Data from device to host
    cudaMemcpy(C_h, C_d, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto end = steady_clock::now();
    
    #ifdef DEBUG
        cout << "Elapsed time in milliseconds: " << duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
        cout << "Elapsed kernel time in microseconds: " << duration_cast<chrono::microseconds>(kernel_end - kernel_start).count() << " μs" << endl;


        if(N < 32 && M < 32 && K < 32) {
            for(int i {0}; i < N ; i++) {
                for(int j {0}; j < M; j++)
                    std::cout << "A[" << i << "][" << j << "] = " << A_h[i * M + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
            
            for(int i {0}; i < M ; i++) {
                for(int j {0}; j < K; j++)
                    std::cout << "B[" << i << "][" << j << "] = " << B_h[i * K + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;

            for(int i {0}; i < N ; i++) {
                for(int j {0}; j < K; j++)
                    std::cout << "C[" << i << "][" << j << "] = " << C_h[i * K + j] << " ";
                std::cout << std::endl;
            }
        }
        
        for(int i {0}; i < N ; i++) 
            for(int j {0}; j < K; j++)
                if(C_h[i * K + j] != M) {
                    cout << "Error on element (" << i << "," << j  << ") = " << C_h[i * K + j] << endl;
                    break;
                }
    #endif

    #ifndef DEBUG
        cout << duration_cast<chrono::milliseconds>(end - start).count() << ", " << duration_cast<chrono::microseconds>(kernel_end - kernel_start).count() << "";
    #endif
    
    
    // Free device memory
    cudaFree(A_d);     
    cudaFree(B_d);
    cudaFree(C_d);

    // Destroy cublas context
    cublasDestroy(handle);  
    
    // Free host memory
    free(A_h);      
    free(B_h);
    free(C_h);

    return EXIT_SUCCESS;
}


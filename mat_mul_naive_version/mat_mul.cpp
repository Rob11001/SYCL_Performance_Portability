#include <iostream>
#include <CL/cl.h>
#include <CL/sycl.hpp>

#define DEBUG
#define SELECTOR 1 // 1 for GPU, 0 for CPU
#define BLOCK_SIZE 4

using namespace sycl;

/**
 * @brief Mat Mul
*/

int main(int argc, char **argv) {
    size_t N, M, K;
    
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <M> <K>" << std::endl;

        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);

    // Allocate matrix (see if can be use C++ classes)
    float *A = static_cast<float *>(malloc(sizeof(float) * N * M));
    float *B = static_cast<float *>(malloc(sizeof(float) * M * K));
    float *C = static_cast<float *>(malloc(sizeof(float) * N * K));

    // Initialization
    for(int i {0}; i < N * M; i++)
        A[i] = 1.0f; // rand() % 5;
    
    for(int i {0}; i < M * K; i++)
        B[i] = 1.0f; //rand() % 5;
    
    for(int i {0}; i < N * K; i++)
        C[i] = 0.0f;


    // Get the queue 
    queue myQueue { 
        #if SELECTOR
            gpu_selector()  
        #else 
            host_selector() 
        #endif
    };
    
    // Use of RAII
    {
        buffer<float, 1> A_buf {A, N * M};
        buffer<float, 1> B_buf {B, M * K};
        buffer<float, 1> C_buf {C, N * K};

        myQueue.submit([&] (handler& cgh) {
            
            accessor A_acc {A_buf, cgh, read_only};
            accessor B_acc {B_buf, cgh, read_only};
            accessor C_acc {C_buf, cgh, write_only, no_init};
            
            range local {BLOCK_SIZE, BLOCK_SIZE};
            range global {K, N};
            accessor<float, 2, access::mode::read_write, access::target::local> tileA {local, cgh};
            accessor<float, 2, access::mode::read_write, access::target::local> tileB {local, cgh};
            
            // TODO: debug
            cgh.parallel_for(nd_range{global, local}, [=] (nd_item<2> it) {
                int bx = it.get_group(0);
                int by = it.get_group(1);
                
                int tx = it.get_local_id(0);
                int ty = it.get_local_id(1);

                int aBegin = M * BLOCK_SIZE * by;
                int aEnd = aBegin + M - 1;
                int aStep = BLOCK_SIZE;

                int bBegin = BLOCK_SIZE * bx;
                int bStep = BLOCK_SIZE * K;
                
                float Csub = 0.0f;
                for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
                    tileA[ty][tx] = A_acc[a + M * ty + tx];
                    tileB[ty][tx] = B_acc[b + K * ty + tx];

                    it.barrier(access::fence_space::local_space);

                    for(int k = 0; k < BLOCK_SIZE; k++)
                        Csub += tileA[ty][k] * tileB[k][tx];

                    it.barrier(access::fence_space::local_space);
                }

                int x = it.get_global_id(0);
                int y = it.get_global_id(1);
                // Writes in global memory
                C_acc[x + y * K] = Csub;
            }); 
          
        }); 
    }


    #ifdef DEBUG
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
        
    #endif

    // Deallocate memory
    free(A);
    free(B);
    free(C);

    return 0; 
}
#include <iostream>
#include <CL/sycl.hpp>

#define DEBUG
#define SELECTOR 1 // 1 for GPU, 0 for CPU
#define BLOCK_SIZE 4

using namespace cl::sycl;

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
        A[i] = rand() % 5;
    
    for(int i {0}; i < M * K; i++)
        B[i] = rand() % 5;
    
    for(int i {0}; i < N * K; i++)
        C[i] = 0.0f;
    
    // Use of RAII
    {
        
        try {
            // Get the queue 
            queue myQueue { 
                #if SELECTOR
                    gpu_selector()  
                #else 
                    cpu_selector() 
                #endif
            };

            buffer<float, 1> A_buf {A, N * M};
            buffer<float, 1> B_buf {B, M * K};
            buffer<float, 1> C_buf {C, N * K};

            myQueue.submit([&] (handler& cgh) {
                
                accessor A_acc {A_buf, cgh, read_only};
                accessor B_acc {B_buf, cgh, read_only};
                accessor C_acc {C_buf, cgh, write_only, no_init};
                
                range local {BLOCK_SIZE, BLOCK_SIZE};
                range global {N, K};
                
                cgh.parallel_for(nd_range{global, local}, [=] (nd_item<2> it) {
                    int row = it.get_global_id(0);
                    int col = it.get_global_id(1); 
                
                    // Each thread calculate an element of the C matrix
                    float acc = 0;
                    #pragma unroll
                    for (size_t i = 0; i < M; i++) {
                        acc += A_acc[i + row * M] * B_acc[col + i * K]; // Reads from global memory
                    }

                    // Writes in global memory
                    C_acc[col + row * K] = acc;
                }); 
            
            });

        } catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
         
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
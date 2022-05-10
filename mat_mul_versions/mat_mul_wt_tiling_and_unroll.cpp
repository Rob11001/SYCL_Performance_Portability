#include <iostream>
#include <CL/sycl.hpp>

#ifndef SELECTOR
    #define SELECTOR 1 // 1 for GPU, 0 for CPU
#endif

#ifndef TILE_SIZE
    #define TILE_SIZE 4
#endif

#ifndef UNROLL_STEP_SIZE
    #define UNROLL_STEP_SIZE 1
#endif

using namespace cl::sycl;
using namespace std::chrono;

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
    
    // Use of RAII
    auto start = steady_clock::now();

    uint64_t start_time, end_time;
    event e;

    {
        // Get the queue 
        queue myQueue { 
            #if SELECTOR
                gpu_selector()  
            #else 
                host_selector() 
            #endif
            , 
            { property::queue::enable_profiling() }
        };

        start = steady_clock::now();
        buffer<float, 1> A_buf {A, N * M};
        buffer<float, 1> B_buf {B, M * K};
        buffer<float, 1> C_buf {C, N * K};

        try {
            e = myQueue.submit([&] (handler& cgh) {
                accessor A_acc {A_buf, cgh, read_only};
                accessor B_acc {B_buf, cgh, read_only};
                accessor C_acc {C_buf, cgh, write_only, no_init};
                
                range local {TILE_SIZE, TILE_SIZE};
                range global {N, K};
                local_accessor<float, 2> tileA {local, cgh};
                local_accessor<float, 2> tileB {local, cgh};
                
                // TODO: debug -> work only when C matrix dimensions are multiple of TILE_SIZE
                cgh.parallel_for(nd_range{global, local}, [=] (nd_item<2> it) {
                    // Global index
                    int x = it.get_global_id(0);
                    int y = it.get_global_id(1);

                    // Group index
                    int bx = it.get_group(0);
                    int by = it.get_group(1);
                    
                    // Local index in the work-group
                    int tx = it.get_local_id(0);
                    int ty = it.get_local_id(1);

                    // Index of the first tile to be processed
                    int aBegin = M * TILE_SIZE * bx;
                    // Index of the last tile of A matrix to be processed
                    int aEnd = aBegin + M - 1;
                    // Step size
                    int aStep = TILE_SIZE;

                    // Index of the first tile of B matrix to be processed
                    int bBegin = TILE_SIZE * by;
                    // Step size
                    int bStep = TILE_SIZE * K;
                    
                    float Csub = 0.0f;
                    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
                        // Load the tile in the local memory (each thread loads one element of A and one element of B)
                        tileA[tx][ty] = A_acc[a + M * tx + ty];
                        tileB[tx][ty] = B_acc[b + K * tx + ty];
            
                        it.barrier(access::fence_space::local_space);
                        
                        // Each thread computes one element using the loaded tile
                        
                        #pragma unroll UNROLL_STEP_SIZE
                        for(int k = 0; k < TILE_SIZE; k++)
                            Csub += tileA[tx][k] * tileB[k][ty];
                        
                        
                        it.barrier(access::fence_space::local_space);
                    }

                    // Writes in global memory
                    C_acc[y + x * K] = Csub;
                });
            
            });
            myQueue.wait_and_throw();
        } catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
            // Deallocate memory
            free(A);
            free(B);
            free(C);
            
            return EXIT_FAILURE;
        }
       
    }

    auto end = steady_clock::now();
    e.wait();
    end_time = e.get_profiling_info<
            cl::sycl::info::event_profiling::command_end>();
    start_time = e.get_profiling_info<
            cl::sycl::info::event_profiling::command_start>();

     #ifdef DEBUG
        std::cout << "Elapsed time in milliseconds: " << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;
        std::cout << "Elapsed kernel time in microseconds: " << ((end_time - start_time) / 1.0e3 )<< " μs" << std::endl;

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

        for(int i {0}; i < N ; i++) 
            for(int j {0}; j < K; j++)
                if(C[i * K + j] != M) {
                    std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                    i = N;
                    break;
                }

    #endif

    #ifndef DEBUG
        std::cout << duration_cast<milliseconds>(end - start).count() << ", " << ((end_time - start_time) / 1.0e3 ) << "";
    #endif

    // Deallocate memory
    free(A);
    free(B);
    free(C);

    return 0; 
}
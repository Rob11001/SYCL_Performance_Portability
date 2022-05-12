#include <iostream>
#include <CL/sycl.hpp>

#ifndef SELECTOR
    #define SELECTOR 1 // 1 for GPU, 0 for CPU
#endif

#ifndef TILE_SIZE
    #define TILE_SIZE 4
#endif

using namespace cl::sycl;
using namespace std::chrono;

/**
 * @brief Mat Mul
*/

// Kernel class
template<int tile_size>
class MatMulKernel {
    private:
        size_t N, M, K;
        accessor<float, 1, access_mode::read> A_acc;
        accessor<float, 1, access_mode::read> B_acc;
        accessor<float, 1, access_mode::write> C_acc;
        local_accessor<float, 2> tileA;
        local_accessor<float, 2> tileB;
    
    public:
        MatMulKernel(const accessor<float, 1, access_mode::read>& A_acc, const accessor<float, 1, access_mode::read>& B_acc, const accessor<float, 1, access_mode::write>& C_acc, const size_t& N, const size_t& M, const size_t& K, const local_accessor<float, 2>& tileA, local_accessor<float, 2>& tileB):
            A_acc(A_acc), B_acc(B_acc), C_acc(C_acc), N(N), M(M), K(K), tileA(tileA), tileB(tileB) {}

        void operator()(nd_item<2> it) const {
            // Group index
            int bx = it.get_group(0);
            int by = it.get_group(1);
            
            // Local index in the work-group
            int tx = it.get_local_id(0);
            int ty = it.get_local_id(1);

            // Global index
            int x = bx * 2 * tile_size + tx;
            int y = by * 2 * tile_size + ty;

            // Index of the first tile to be processed
            int aBegin = 2 * M * tile_size * bx;
            // Index of the last tile of A matrix to be processed
            int aEnd = aBegin + M - 1;
            // Step size
            int aStep = 2 * tile_size;
            // Index of the first tile of B matrix to be processed
            int bBegin = 2 * tile_size * by;
            // Step size
            int bStep = 2 * tile_size * K;
            
            float Csub_1 = 0.0f, Csub_2 = 0.0f, Csub_3 = 0.0f, Csub_4 = 0.0f;
            for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
                // Load the tile in the local memory (each thread loads one element of A and one element of B)
                tileA[tx][ty] = A_acc[a + M * tx + ty];
                tileA[tx][ty + tile_size] = A_acc[a + M * tx + ty + tile_size];
                tileA[tx + tile_size][ty] = A_acc[a + M * tx + ty + tile_size * M];
                tileA[tx + tile_size][ty + tile_size] = A_acc[a + M * tx + ty + tile_size + tile_size * M];

                tileB[tx][ty] = B_acc[b + K * tx + ty];
                tileB[tx][ty + tile_size] = B_acc[b + K * tx + ty + tile_size];
                tileB[tx + tile_size][ty] = B_acc[b + K * tx + ty + tile_size * K];
                tileB[tx + tile_size][ty + tile_size] = B_acc[b + K * tx + ty + tile_size + tile_size * K];

                it.barrier(access::fence_space::local_space);
                
                // Each thread computes one element using the loaded tile
                for(int k = 0; k < tile_size * 2 ; k++) {
                    Csub_1 += tileA[tx][k] * tileB[k][ty];
                    Csub_2 += tileA[tx][k] * tileB[k][ty + tile_size];
                    Csub_3 += tileA[tx + tile_size][k] * tileB[k][ty];
                    Csub_4 += tileA[tx + tile_size][k] * tileB[k][ty + tile_size];
                }
                
                it.barrier(access::fence_space::local_space);
            }
            // Writes in global memory
            C_acc[y + x * K] = Csub_1;
            C_acc[y + x * K + tile_size] = Csub_2; 
            C_acc[y + x * K + tile_size * K] = Csub_3;
            C_acc[y + x * K + tile_size + tile_size * K] = Csub_4; 
        }
};


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
        A[i] = (i % 2);
    
    for(int i {0}; i < M * K; i++)
        B[i] = (i + 1) % 2;
    
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
                range global {N/2, K/2};
                local_accessor<float, 2> tileA {range {2 * TILE_SIZE, TILE_SIZE * 2}, cgh};
                local_accessor<float, 2> tileB {range {2 * TILE_SIZE, TILE_SIZE * 2}, cgh};
                
                // TODO: debug -> work only when C matrix dimensions are multiple of TILE_SIZE
                cgh.parallel_for(nd_range{global, local}, MatMulKernel<TILE_SIZE>(A_acc, B_acc, C_acc, N, M, K, tileA, tileB));
            
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
                if(C[i * K + j] != ((j + 1) % 2) * (M/2)) {
                    std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                    i = N;
                    break;
                }

    #endif

    #ifndef DEBUG
        for(int i {0}; i < N ; i++) 
            for(int j {0}; j < K; j++)
                if(C[i * K + j] != ((j + 1) % 2) * (M/2)) {
                    std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                    i = N;
                    break;
                }
        std::cout << duration_cast<milliseconds>(end - start).count() << ", " << ((end_time - start_time) / 1.0e3 ) << "";
    #endif

    // Deallocate memory
    free(A);
    free(B);
    free(C);

    return 0; 
}
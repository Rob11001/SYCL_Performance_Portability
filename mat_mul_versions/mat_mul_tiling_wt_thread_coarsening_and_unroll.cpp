#include <iostream>
#include <CL/sycl.hpp>

#define MIN(a,b) (((a)<(b))?(a):(b))

#ifndef SELECTOR
    #define SELECTOR 1 // 1 for GPU, 0 for CPU
#endif

#ifndef TILE_SIZE
    #define TILE_SIZE 4
#endif

#ifndef C_FACTOR_X
    #define C_FACTOR_X 2
#endif

#ifndef C_FACTOR_Y
    #define C_FACTOR_Y 2
#endif

using namespace cl::sycl;
using namespace std::chrono;

/**
 * @brief Mat Mul
*/

// Kernel class
template<int tile_size, int coarse_factor_x, int coarse_factor_y>
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
            N(N), M(M), K(K), A_acc(A_acc), B_acc(B_acc), C_acc(C_acc), tileA(tileA), tileB(tileB) {}

        void operator()(nd_item<2> it) const {
            // Group index
            int bx = it.get_group(0);
            int by = it.get_group(1);
            
            // Local index in the work-group
            int tx = it.get_local_id(0) * coarse_factor_x;
            int ty = it.get_local_id(1) * coarse_factor_y;

            // Global index
            int x = bx * (tile_size) + tx;
            int y = by * (tile_size) + ty;

            // Index of the first tile to be processed
            int aBegin = M * (tile_size) * bx;
            // Index of the last tile of A matrix to be processed
            int aEnd = aBegin + M - 1;
            // Step size
            int aStep = tile_size;
            // Index of the first tile of B matrix to be processed
            int bBegin = (tile_size) * by;
            // Step size
            int bStep = (tile_size) * K;
            
            float Csub[coarse_factor_x][coarse_factor_y] {};
            for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {    
                // Load the tile in the local memory (each thread loads coarse_factor x coarse_factor elements from A and from B)
                #pragma unroll
                for(int i {0}; i < coarse_factor_x; i++)
                    #pragma unroll
                    for(int j {0}; j < coarse_factor_y; j++) {
                        tileA[tx + i][ty + j] = A_acc[a + M * tx + ty + i * M + j];
                        tileB[tx + i][ty + j] = B_acc[a + K * tx + i * K + ty + j ];
                    }

                it.barrier(access::fence_space::local_space);
               
                // Each thread computes coarse_factor elements using the loaded tile
                #ifndef UNROLL_STEP_SIZE 
                    #pragma unroll
                #else
                    #pragma unroll UNROLL_STEP_SIZE 
                #endif
                for(int k = 0; k < tile_size; k++)
                    #pragma unroll
                    for(int i {0}; i < coarse_factor_x; i++)
                        #pragma unroll
                        for(int j {0}; j < coarse_factor_y; j++) {
                            Csub[i][j] += tileA[tx + i][k] * tileB[k][ty + j];
                        }
               
                it.barrier(access::fence_space::local_space);
            }

            // Writes in global memory the coarse_factor elements that thread has computed
            int baseline = y + x * K;
            #pragma unroll
            for(int i {0}; i < coarse_factor_x; i++)
                #pragma unroll
                for(int j {0}; j < coarse_factor_y; j++) {
                    C_acc[baseline + i * K + j] = Csub[i][j];
                }
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
    for(size_t i {0}; i < N * M; i++)
        A[i] = (i % 2);
    
    for(size_t i {0}; i < M * K; i++)
        B[i] = (i + 1) % 2;
    
    for(size_t i {0}; i < N * K; i++)
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
                
                range local {TILE_SIZE / C_FACTOR_X, TILE_SIZE / C_FACTOR_Y};
                range global {N / C_FACTOR_X, K / C_FACTOR_Y};
                local_accessor<float, 2> tileA {range {TILE_SIZE, TILE_SIZE}, cgh};
                local_accessor<float, 2> tileB {range {TILE_SIZE, TILE_SIZE}, cgh};
                
                // REMEMBER: work only when C matrix dimensions are multiple of TILE_SIZE
                cgh.parallel_for(nd_range{global, local}, MatMulKernel<TILE_SIZE, C_FACTOR_X, C_FACTOR_Y>(A_acc, B_acc, C_acc, N, M, K, tileA, tileB));
            
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
            for(size_t i {0}; i < N ; i++) {
                for(size_t j {0}; j < M; j++)
                    std::cout << "A[" << i << "][" << j << "] = " << A[i * M + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
            
            for(size_t i {0}; i < M ; i++) {
                for(size_t j {0}; j < K; j++)
                    std::cout << "B[" << i << "][" << j << "] = " << B[i * K + j] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;

            for(size_t i {0}; i < N ; i++) {
                for(size_t j {0}; j < K; j++)
                    std::cout << "C[" << i << "][" << j << "] = " << C[i * K + j] << " ";
                std::cout << std::endl;
            }
        }

        for(size_t i {0}; i < N ; i++) 
            for(size_t j {0}; j < K; j++)
                if(C[i * K + j] != ((j + 1) % 2) * (M/2)) {
                    std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                    i = N;
                    break;
                }

    #endif

    #ifndef DEBUG
        #ifndef TEST
            for(size_t i {0}; i < N ; i++) 
                for(size_t j {0}; j < K; j++)
                    if(C[i * K + j] != ((j + 1) % 2) * (M/2)) {
                        std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                        i = N;
                        break;
                    }
            std::cout << duration_cast<milliseconds>(end - start).count() << ", " << ((end_time - start_time) / 1.0e3 ) << "";
        #endif
    #endif

    #ifdef TEST
        for(int i {0}; i < N ; i++) 
            for(int j {0}; j < K; j++)
                if(C[i * K + j] != ((j + 1) % 2) * (M/2)) {
                    std::cout << "Error: (" << i << ", " << j << "): " << C[i * K + j] << std::endl;
                    i = N;
                    break;
                }
        std::cout << duration_cast<milliseconds>(end - start).count() << " ";
    #endif

    // Deallocate memory
    free(A);
    free(B);
    free(C);

    return 0; 
}
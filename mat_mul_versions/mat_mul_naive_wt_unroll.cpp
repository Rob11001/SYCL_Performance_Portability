#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>

#ifndef SELECTOR
    #define SELECTOR 1 // 1 for GPU, 0 for CPU
#endif

#ifndef BLOCK_SIZE_X
    #define BLOCK_SIZE_X 4
#endif

#ifndef BLOCK_SIZE_Y
    #define BLOCK_SIZE_Y 4
#endif

using namespace cl::sycl;
using namespace std::chrono;

/**
 * @brief Mat Mul
*/

// Kernel class
class MatMulKernel {
    private:
        size_t N, M, K;
        accessor<float, 1, access_mode::read> A_acc;
        accessor<float, 1, access_mode::read> B_acc;
        accessor<float, 1, access_mode::write> C_acc;
    
    public:
        MatMulKernel(const accessor<float, 1, access_mode::read>& A_acc, const accessor<float, 1, access_mode::read>& B_acc, const accessor<float, 1, access_mode::write>& C_acc, const size_t& N, const size_t& M, const size_t& K):
            A_acc(A_acc), B_acc(B_acc), C_acc(C_acc), N(N), M(M), K(K) {}

        void operator()(nd_item<2> it) const {
            int row = it.get_global_id(0);
            int col = it.get_global_id(1); 
        
            // Each thread calculate an element of the C matrix
            float acc = 0;
            #ifndef UNROLL_STEP_SIZE 
                #pragma unroll
            #else
                #pragma unroll UNROLL_STEP_SIZE 
            #endif
            for (size_t i = 0; i < M; i++) 
                acc += A_acc[i + row * M] * B_acc[col + i * K]; // Reads from global memory
            
            // Writes in global memory
            C_acc[col + row * K] = acc;
            
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
        A[i] = 1.0f; //rand() % 5;
    
    for(int i {0}; i < M * K; i++)
        B[i] = 1.0f; //rand() % 5;
    
    for(int i {0}; i < N * K; i++)
        C[i] = 0.0f;
    
    // Use of RAII
    auto start = steady_clock::now();

    uint64_t start_time, end_time, start_submit;
    event e;
    {
        
        try {
            // Get the queue 
            queue myQueue { 
                #if SELECTOR
                    gpu_selector()  
                #else 
                    cpu_selector() 
                #endif
            ,
                { property::queue::enable_profiling() }
            };

            start = steady_clock::now();

            buffer<float, 1> A_buf {A, N * M};
            buffer<float, 1> B_buf {B, M * K};
            buffer<float, 1> C_buf {C, N * K};

            e = myQueue.submit([&] (handler& cgh) {
                
                accessor A_acc {A_buf, cgh, read_only};
                accessor B_acc {B_buf, cgh, read_only};
                accessor C_acc {C_buf, cgh, write_only, no_init};
                
                range local {BLOCK_SIZE_X, BLOCK_SIZE_Y};
                range global {N, K};
                
                cgh.parallel_for(nd_range{global, local}, MatMulKernel(A_acc, B_acc, C_acc, N, M, K)); 
            });
            
            myQueue.wait_and_throw();
            
        } catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
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
        #ifndef TEST
            for(int i {0}; i < N ; i++) 
                for(int j {0}; j < K; j++)
                    if(C[i * K + j] != M) {
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
                if(C[i * K + j] != M) {
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
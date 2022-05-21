import os
import sys
sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
import hypermapper

files_to_process = ["mat_mul_naive"] 
all_files = ["mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_naive_wt_coarsening", "mat_mul_naive_wt_coarsening_and_unroll",
    "mat_mul_tiling", "mat_mul_tiling_wt_unroll", "mat_mul_tiling_wt_thread_coarsening", "mat_mul_tiling_wt_thread_coarsening_and_unroll"]
size = [4096, 8192]
device = ["CPU", "GPU"]
n_test = 5
file = " "
selector = 1

def mat_mul(X):
    command = "syclcc -O3 ../{0}.cpp -o ../{1}.out -DTEST -DSELECTOR={2}".format(file, file, selector)
    if "tiling" in file:
        tile_size = X['tile_size']
        command = "{0} -DTILE_SIZE={1}".format(command, tile_size)
    else:
        block_size_x = X['block_size_x']
        block_size_y = X['block_size_y']
        command = "{0} -DBLOCK_SIZE_X={1} -DBLOCK_SIZE_Y={2}".format(command, block_size_x, block_size_y)
    
    if "coarsening" in file:
        coarse_factor_x = X['coarse_factor_x']
        coarse_factor_y = X['coarse_factor_y']
        command = "{0} -DC_FACTOR_X={1} -DC_FACTOR_Y={2}".format(command, coarse_factor_x, coarse_factor_y)

    if "unrolling" in file:
        unroll_step = X['unroll_step']
        if unroll_step != 0:
            command = "{0} -DUNROLL_STEP_SIZE={1}".format(command, unroll_step)  

    os.system(command)

    time = 0
    for i in range(n_test):
        str_time = os.popen("./../{0}.out {1} {1} {1}".format(file, size[selector])).read()
        if "Error" not in str_time:
            time +=  int(str_time)
        else:
            time = sys.maxsize
            break 

    return time / n_test


for name in files_to_process:
    file = name
    json_path = "./json/{0}/{1}.json".format(device[selector], name)
    hypermapper.optimizer.optimize(json_path, mat_mul)
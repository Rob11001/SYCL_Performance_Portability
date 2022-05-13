#!/usr/bin/env python
import os

files = ("mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_naive_wt_coarsening", "mat_mul_naive_wt_coarsening_and_unroll", "mat_mul_tiling", "mat_mul_tiling_wt_unroll", "mat_mul_tiling_wt_coarsening", "mat_mul_tiling_wt_coarsening_and_unroll")
sizes = ("1024 1024 1024","2048 2048 2048", "4096 4096 4096", "8192 8192 8192")
tile_sizes = [4, 8, 16, 32]
step_sizes = [0, 1, 4, 8, 16, 32, 64]
coarse_factors = [2, 4, 8]
tests = 1
csv_name = "test"
device_type = (1, 0)
device_name = {0: "CPU", 1: "GPU"}
device_flag = {1: "cuda:sm_86", 0: "omp"}

start_step = {
    "device": 0,
    "file": 0,
    "tile_size": 0
}

current_step = {}
current_step_filename = "log.txt"
current_step_file = open(current_step_filename, mode="w")

for d_step in range(start_step["device"], len(device_type), 1):    
    device = device_type[d_step]
    current_step["device"] = d_step;

    print("export HIPSYCL_TARGETS={}".format(device_flag[device]))
    os.environ["HIPSYCL_TARGETS"] = device_flag[device];
    
    
    print("Tests {0}\n".format(device_name[device]))
    for t_step in range(start_step["tile_size"], len(tile_sizes), 1):
        tile_size = tile_sizes[t_step]
        current_step["tile_size"] = t_step;
        # Tests
        for f_step in range(start_step["file"], len(files), 1):
            file = files[f_step]
            current_step["file"] = f_step

            current_step_file.write("device: {0}, file: {1}, tile_size: {2}\n".format(current_step["device"], current_step["file"], current_step["tile_size"]))

            need_unroll_tuning = "unroll" in file
            need_coarsening_tuning = "coarsening" in file
            unrolling_iterations = range(len(step_sizes)) if need_unroll_tuning else range(1)
            coarsening_iterations = range(len(coarse_factors)) if need_coarsening_tuning else range(1)
            # Iterations for unroll tuning
            for i in unrolling_iterations:
                for j in coarsening_iterations:
                    # Files compilation
                    print("Compiling...")
                    command = ""
                    if need_unroll_tuning:
                        if i != 0:
                            command = "syclcc -O3 {0}.cpp -o {1}.out -DTILE_SIZE={2} -DSELECTOR={3} -DUNROLL_STEP_SIZE={4}".format(file, file, tile_size, device, step_sizes[i])
                        else:
                            command = "syclcc -O3 {0}.cpp -o {1}.out -DTILE_SIZE={2} -DSELECTOR={3}".format(file, file, tile_size, device)
                    else:
                        command = "syclcc -O3 {0}.cpp -o {1}.out -DTILE_SIZE={2} -DSELECTOR={3}".format(file, file, tile_size, device)
                    
                    if need_coarsening_tuning:
                        command = "{0} -DC_FACTOR={1}".format(command, coarse_factors[j])
                    
                    os.system(command)
                    
                    print("done\n")
                    
                    csv_filename = "{0}_{1}_{2}_{3}".format(csv_name, file, device_name[device], tile_size)
                    if need_unroll_tuning:
                        csv_filename = "{0}_{1}".format(csv_filename, step_sizes[i])
                    if need_coarsening_tuning:
                        csv_filename = "{0}_{1}".format(csv_filename, coarse_factors[j])
                    csv_filename += ".csv"
                    csv_file = open("{0}/{1}".format("tests", csv_filename), mode="w")
                
                    print("Creating csv file: {}".format(csv_filename))
                    # headers
                    header = "NxMxK, "
                    for i in range(tests):
                        header += "t{0}, k{0}, ".format(i + 1)
                    
                    csv_file.write("{0}\n".format(header[0: len(header) - 2]))

                    # Tests
                    print("Running tests: {}".format(file))
                    for size in sizes:
                        test_line = "{0}, ".format(size)

                        for i in range(tests):
                            print("./{0}.out {1}".format(file, size))
                            times = os.popen("./{0}.out {1}".format(file, size)).read()
                            if "Error" in times:
                                times = "N/A, N/A"
                            # Controllare se times contiene "Error:"
                            test_line += "{0}, ".format(times)
                        csv_file.write("{0}\n".format(test_line[0: len(test_line) - 2]))
                    print("Done")
        # reset
        start_step["file"] = 0
    # reset
    start_step["tile_size"] = 0

                    


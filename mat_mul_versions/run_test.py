#!/usr/bin/env python
import os

files = ("mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_wt_tiling", "mat_mul_wt_tiling_and_unroll")
sizes = ("1024 1024 1024", "2048 2048 2048") #, "4096 4096 4096", "8192 8192 8192")
tile_sizes = [4, 8, 16, 32, 64, 128]
tests = 5
csv_name = "test"
device_type = (0, 1)
device_name = {0: "CPU", 1: "GPU"}
device_flag = {1: "cuda:sm_86", 0: "omp"}

for device in device_type:    
    print("export HIPSYCL_TARGETS={}".format(device_flag[device]))
    os.environ["HIPSYCL_TARGETS"] = device_flag[device];
    
    
    print("Tests {0}\n".format(device_name[device]))
    for tile_size in tile_sizes:
        # Tests
        for file in files:
            # Files compilation
            print("Compiling...")
            os.system("syclcc -O3 {0}.cpp -o {1} -DTILE_SIZE={2} -DSELECTOR={3}".format(file, file, tile_size, device))
            print("done\n")
            
            csv_filename = "{0}_{1}_{2}_{3}.csv".format(csv_name, file, device_name[device], tile_size)
            csv_file = open(file=csv_filename, mode="w")
        
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
                    print("./{0} {1}".format(file, size))
                    times = os.popen("./{0} {1}".format(file, size)).read()
                    test_line += "{0}, ".format(times)
                csv_file.write("{0}\n".format(test_line[0: len(test_line) - 2]))
            print("Done")
                


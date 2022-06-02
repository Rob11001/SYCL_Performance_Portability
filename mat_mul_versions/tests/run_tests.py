# Script that runs the test using the parameters found by the hypermapper (Note: reads the parameters for each version from the '{CPU/GPU}/samples/opt' directory)

import csv
import os

files = ("mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_naive_wt_coarsening", "mat_mul_naive_wt_coarsening_and_unroll", "mat_mul_tiling", "mat_mul_tiling_wt_unroll", "mat_mul_tiling_wt_thread_coarsening", "mat_mul_tiling_wt_thread_coarsening_and_unroll")

devices = ["CPU", "GPU"]
device_flag = {"GPU": "cuda:sm_86", "CPU": "omp"}

sizes = {
    "CPU": ("1024 1024 1024", "2048 2048 2048", "4096 4096 4096"),
    "GPU": ("1024 1024 1024", "2048 2048 2048", "4096 4096 4096", "8192 8192 8192")
}
n_test = 5

for device in devices:
    print("export HIPSYCL_TARGETS={}".format(device_flag[device]))
    os.environ["HIPSYCL_TARGETS"] = device_flag[device];
    
    print("Tests on {0}\n".format(device))
    for file in files:
        filepath = "./{0}/samples/opt/{1}_{0}_output_samples.csv".format(device, file)
        with open(filepath, mode="r") as input, \
            open("./{0}/times/{1}.csv".format(device, file), mode="w") as output:
            
            reader = csv.DictReader(input)
            fieldnames = reader.fieldnames
            fieldnames = ["NxMxK"] + [field for field in fieldnames if field != "Time"]
            for i in range(n_test):
                fieldnames += ["t{0}".format(i), "k{0}".format(i)]
            fieldnames += ["Avg Time", "Avg Kernel Time"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            print("Compiling...")
            command = "syclcc -O3 ../{0}.cpp -o ../{1}.out -DSELECTOR={2}".format(file, file, devices.index(device))

            row = next(reader)
            row.pop("Time")
            if "naive" in file:
                block_size_x = row["block_size_x"]
                block_size_y = row["block_size_y"]
                command = "{0} -DBLOCK_SIZE_X={1} -DBLOCK_SIZE_Y={2}".format(command, block_size_x, block_size_y)
            else:
                tile_size = row["tile_size"]
                command = "{0} -DTILE_SIZE={1}".format(command, tile_size)
            
            if "unroll" in file:
                unroll_step_size = row["unroll_step"]
                if unroll_step_size != "0":
                    command = "{0} -DUNROLL_STEP_SIZE={1}".format(command, unroll_step_size)

            if "coarsening" in file:
                coarse_factor_x = row["coarse_factor_x"]
                coarse_factor_y = row["coarse_factor_y"]
                command = "{0} -DC_FACTOR_X={1} -DC_FACTOR_Y={2}".format(command, coarse_factor_x, coarse_factor_y)
            
            print(command)
            os.system(command)
            print("done\n")

            for size in sizes[device]:            
                avg = 0
                avgKernel = 0
                line = row
                line["NxMxK"] = size
                for test in range(n_test):
                    print("../{0}.out {1}".format(file, size))
                    time = os.popen("../{0}.out {1}".format(file, size)).read()                    
                    [total_time, kernel_time] = time.split(",")
                    line["t{0}".format(test)] = total_time
                    avg += float(total_time)
                    line["k{0}".format(test)] = kernel_time
                    avgKernel += float(kernel_time)
                avg = avg / n_test
                avgKernel = avgKernel / n_test
                line["Avg Time"] = avg
                line["Avg Kernel Time"] = avgKernel
                writer.writerow(line)
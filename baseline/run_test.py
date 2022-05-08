#!/usr/bin/env python
import os

files = ("mat_mul_cublas", "mat_mul_mkl")
sizes = ("1024 1024 1024", "2048 2048 2048", "4096 4096 4096", "8192 8192 8192")
tests = 5
csv_name = "test"

# Files compilation
print("Compiling...")

os.system("nvcc --gpu-architecture=compute_86 {0}.cpp -o {1} -lcublas".format(files[0], files[0]))
os.system("dpcpp {0}.cpp -DMKL_ILP64 -qmkl=parallel -o {1} -O3".format(files[1], files[1]))

print("done")


# Tests
for file in files:
    csv_filename = "{0}_{1}.csv".format(csv_name, file)
    csv_file = open(file=csv_filename, mode="w")
    
    print("Creating csv file: {}".format(csv_filename))
    # headers
    header = "NxMxK, "
    if file == "mat_mul_cublas":
        for i in range(tests):
            header += "t{0}, k{0}, ".format(i + 1)
    else:
        for i in range(tests):
            header += "t{0}, ".format(i + 1) 

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
        


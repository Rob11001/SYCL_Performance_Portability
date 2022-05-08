#!/bin/bash
files=("seq_reduction" "reduction_v1" "reduction_v2" "reduction_v3" "reduction_v4" "reduction_v5")
sizes=(33554432 67108864 134217728 268435456)
tests=5
csv="benchmark"

# Compilazione file C
for file in "${files[@]}"; do
    echo `g++ -o ${file}.exe ${file}.cpp -lOpenCL`;
done

for file in "${files[@]}"; do   
    # Creazione Headers nel file di output
    filename="${csv}_${file}.csv"
    echo "N, t1, k1, t2, k2, t3, k3, t4, k4, t5, k5" >> ${filename}

    for size in "${sizes[@]}"; do
        line="${size}, "
        for ((i=0;i<${tests};i++)); do
            last_time=`./${file}.exe ${size}` 
            line="${line}${last_time}, "  
            echo "Algoritmo {$file} size: ${size} test: ${i} time: ${last_time}"
        done
        end=$(( ${#line} - 2 ))
        echo "${line:0:${end}}" >> ${filename}
    done
done

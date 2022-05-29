import pandas as pd
import plotly.express as px
import csv

files = ("mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_naive_wt_coarsening", "mat_mul_naive_wt_coarsening_and_unroll", "mat_mul_tiling", "mat_mul_tiling_wt_unroll", "mat_mul_tiling_wt_thread_coarsening", "mat_mul_tiling_wt_thread_coarsening_and_unroll", "mat_mul_cublas")

readers = {}
for file in files:
    readers[file] = csv.DictReader( open("./{0}.csv".format(file), mode="r"))


for i in [1024, 2048, 4096, 8192]:
    
    df = pd.DataFrame({
        "Versions": [file[8:len(file)] for file in files],
        "Time (ms)": [float(next(readers[file])["Avg Time"]) for file in files],
    })


    # Plotly Express
    fig = px.bar(df, x="Versions", y="Time (ms)", color="Versions", title="GPU: {0}x{0}".format(i))
    fig.show()


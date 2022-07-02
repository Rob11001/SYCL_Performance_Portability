import math
import pandas as pd
import plotly.io as io
import plotly.express as px
import plotly.graph_objects as go
import csv

files = ("mat_mul_naive", "mat_mul_naive_wt_unroll", "mat_mul_naive_wt_coarsening", "mat_mul_naive_wt_coarsening_and_unroll", "mat_mul_tiling", "mat_mul_tiling_wt_unroll", "mat_mul_tiling_wt_thread_coarsening", "mat_mul_tiling_wt_thread_coarsening_and_unroll", "mat_mul_mkl")
labels = ["Naive", "Naive + unroll", "Naive + coarsening", "Naive + coarsening + unroll", "Tiling", "Tiling + unroll", "Tiling + coarsening", "Tiling + coarsening + unroll", "MKL"]
N = 5
avgTimes = {}
stdDeviation = {}
readers = {}
for file in files:
    readers[file] = csv.DictReader(open("./times/{0}.csv".format(file), mode="r"))

# Computes standard deviation for each file
for i in [1024, 2048, 4096]:
    for file in files:
        id = "{0}{1}".format(file, i)
        line = next(readers[file])
        avgTimes[id] = float(line["Avg Time"])
        stdDeviation[id] = math.sqrt((1/N) * sum([pow(float(line["t{0}".format(j)]) - avgTimes[id], 2) for j in range(N)]))

# Plots all files without standard deviation
for i in [1024, 2048, 4096]:
    df = {
        "<b>Version</b>": ["<b>" + labels[i] + "</b> " for i in range(len(files))],
        "<b>Average Time (ms)</b>": [avgTimes["{0}{1}".format(file, i)] for file in files],
    }

    # Plotly Express
    data = pd.DataFrame(df)
    fig = px.bar(data, y="<b>Version</b>", x="<b>Average Time (ms)</b>", color="<b>Version</b>", title="<b>CPU</b>: {0} x {0}".format(i), text_auto='.2f', orientation="h")
    fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        title = {
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size = 28)
        },
        legend = dict(font = dict(size = 10)),
        legend_title = dict(font = dict(size = 10)),
        yaxis = dict(tickfont=dict(family='Helvetica', size=20, color='black')),
    )
    fig.update_yaxes(title_font=dict(size=16))
    fig.update_xaxes(title_font=dict(size=16))
    fig.update(layout_showlegend=False)
    
    io.write_image(fig, './plots/all_times_{0}.pdf'.format(i), format='pdf', scale=1, width=1400, height=1000)

# Plots only the four best files with their standard deviation
best_files = []
for i in range(len(files)):
    file = files[i]
    id = "{0}{1}".format(file, 4096)
    inserted = False
    for j in range(len(best_files)):
        if avgTimes[id] < float(best_files[j]["time"]):
            best_files.insert(j, {"name": file, "time": avgTimes[id], "label": labels[i]})
            inserted = True
            if len(best_files) > 4:
                best_files.pop()
            break
    if len(best_files) < 4 and not inserted:
        best_files.append({"name": file, "time": avgTimes[id], "label": labels[i]})
              
for i in [1024, 2048, 4096]:
    df = {
        "<b>Version</b>": ["<b>" + file["label"] + "</b> " for file in best_files] + ["<b>" + file["label"] + "</b> " for file in best_files],
        "<b>Average Time (ms)</b>": [avgTimes["{0}{1}".format(file["name"], i)] for file in best_files]  + [stdDeviation["{0}{1}".format(file["name"], i)] for file in best_files],
        " ": ["<b>Avg Time</b>" for file in best_files] + ["<b>Standard Deviation</b>" for file in best_files]
    }

    # Plotly Express
    data = pd.DataFrame(df)
    fig = px.bar(data, y="<b>Version</b>", x="<b>Average Time (ms)</b>", color=" ", title="<b>CPU</b>: {0} x {0}".format(i), text_auto='.2f', orientation="h", barmode="group")
    fig.update_traces(textfont_size=18, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        title = {
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size = 28)
        },
        legend = dict(font = dict(size = 18)),
        yaxis = dict(tickfont=dict(family='Helvetica', size=20, color='black'))
    )
    fig.update_yaxes(title_font=dict(size=16))
    fig.update_xaxes(title_font=dict(size=16))
    
    io.write_image(fig, './plots/best_{0}.pdf'.format(i), format='pdf', scale=1, width=1400, height=1000)
    #fig.show()

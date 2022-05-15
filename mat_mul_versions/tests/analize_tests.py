import csv
from os import listdir
from os.path import isfile, join

devices = ["CPU", "GPU"]
tests = 5
best_times = {
    "CPU": [],
    "GPU": []
}

def update_time(best_times, time, filename, device):
    if len(best_times[device]) < 10:
        best_times[device] = best_times[device] + [{'name': filename, 'time': time}]
    else:
        if best_times[device][len(best_times[device]) - 1]['time'] > time:
            best_times[device][len(best_times[device]) - 1] = {'name': filename, 'time': time}

    best_times[device] = sorted(best_times[device], key=lambda d: d['time'])
        


for device in devices:
    # Takes all test files in device directory
    testfiles = [f for f in listdir("./{}/".format(device)) if isfile(join("./{}/".format(device), f))]

    for file in testfiles:
        filepath = "./{0}/{1}".format(device, file)
        with open(filepath, mode="r") as inputCsvFile, \
            open("./refined_tests/{0}".format(filepath), mode="w") as outputCsvFile:
            reader = csv.DictReader(inputCsvFile)
            fieldnames = reader.fieldnames
            fieldnames = fieldnames + ["Avg time", "Avg kernel time"]
            writer = csv.DictWriter(outputCsvFile, fieldnames=fieldnames)
            writer.writeheader()
            
            # for each line add avg time and avg kernel time on the csv
            for line in reader:
                avg = 0
                avgKernel = 0
                for i in range(tests):
                    tmp = line[" t{0}".format(i + 1)] if 'N/A' not in line[" t{0}".format(i + 1)] else "0"
                    avg += int(tmp)
                    tmp = line[" k{0}".format(i + 1)] if 'N/A' not in line[" k{0}".format(i + 1)] else "0"
                    avgKernel += float(tmp)
                
                if avg == 0:
                    avg = "N/A"
                    avgKernel = "N/A"
                else:
                    avg = avg / tests
                    avgKernel = avgKernel / tests

                row = {}
                for field in reader.fieldnames:
                    row[field] = line[field]
                row['Avg time'] = avg
                row['Avg kernel time'] = avgKernel
                writer.writerow(row)
                
                if avg != 'N/A' and (line['NxMxK'] == "8192 8192 8192" or (device == 'CPU' and line['NxMxK'] == "4096 4096 4096")): 
                    update_time(best_times, avg, filepath, device)


for device in devices:
    print("{0}:".format(device))
    for i in range(10):
        print("{0}, time: {1}".format(best_times[device][i]['name'], best_times[device][i]['time']))

            
            
            

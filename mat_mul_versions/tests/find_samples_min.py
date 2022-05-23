import csv
import sys
from os import listdir
from os.path import isfile, join

devices = ["CPU", "GPU"]

for device in devices:
    samplefiles = [f for f in listdir("./{}/samples/".format(device)) if isfile(join("./{}/samples".format(device), f))]

    for file in samplefiles:
        filepath = "./{0}/samples/{1}".format(device, file)
        with open(filepath, mode="r") as input, \
            open("./{0}/samples/opt/{1}".format(device, file), mode="w") as output:

            reader = csv.DictReader(input)
            fieldnames = reader.fieldnames
            fieldnames = [field for field in fieldnames if field != "Timestamp"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            opt = sys.maxsize
            optline = ""
            for line in reader:
                time = line["Time"]
                time = float(time)
                if time < opt:
                    opt = time
                    optline = line
            optline.pop("Timestamp", None)
            writer.writerow(optline)
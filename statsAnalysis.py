from statistics import *
import os
import csv


def outputStatistics(data: list, metric: str, outputDir: str):

    # get non-zero data following the (+1 data / -1 result) approach
    nonZeroData = [value + 1 for value in data]

    # calculate and output
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")

        
        outputValue(w, metric, "pstdev", pstdev(data))
        


def outputValue(w, metric: str, name: str, value, reduceByOne=False):
    name = "{0}_{1}".format(metric, name)
    value = value - 1 if reduceByOne else value
    w.writerow([name, value])

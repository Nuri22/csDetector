from statistics import *
import os
import csv


def outputStatistics(data: list, metric: str, outputDir: str):

    # validate
    if len(data) < 1:
        return

    # calculate and output
    stats = calculateStats(data)

    # output
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")

        for key in stats:
            outputValue(w, metric, key, stats)


def calculateStats(data):

    # get non-zero data following the (+1 data / -1 result) approach
    nonZeroData = [value + 1 for value in data]

    stats = dict(
        
        stdev=stdev(data) if len(data) > 1 else None,
    )

    return stats


def outputValue(w, metric: str, name: str, dict: dict):
    value = dict[name]
    name = "{0}_{1}".format(metric, name)
    w.writerow([name, value])

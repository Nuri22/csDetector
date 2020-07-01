from statistics import *
import os
import csv


def outputStatistics(data: list, metric: str, outputDir: str):

    # get non-zero data following the (+1 data / -1 result) approach
    nonZeroData = [value + 1 for value in data]

    # calculate and output
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")

        outputValue(w, metric, "min", min(data))
        outputValue(w, metric, "max", max(data))
        outputValue(w, metric, "mean", mean(data))
        outputValue(w, metric, "geometric_mean", geometric_mean(nonZeroData), True)
        outputValue(w, metric, "harmonic_mean", harmonic_mean(data))
        outputValue(w, metric, "median", median(data))
        outputValue(w, metric, "median_low", median_low(data))
        outputValue(w, metric, "median_high", median_high(data))
        outputValue(w, metric, "median_grouped", median_grouped(data))
        outputValue(w, metric, "mode", mode(data))
        # outputValue(w, metric, "multimode", multimode(data))
        # outputValue(w, metric, "quantiles", quantiles(data))
        outputValue(w, metric, "pstdev", pstdev(data))
        outputValue(w, metric, "pvariance", pvariance(data))
        outputValue(w, metric, "stdev", stdev(data))
        outputValue(w, metric, "variance", variance(data))


def outputValue(w, metric: str, name: str, value, reduceByOne=False):
    name = "{0}_{1}".format(metric, name)
    value = value - 1 if reduceByOne else value
    w.writerow([name, value])

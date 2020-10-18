from statistics import *
import os
import csv


def outputStatistics(idx: int, data: list, metric: str, outputDir: str):

    # validate
    if len(data) < 1:
        return

    # calculate and output
    stats = calculateStats(data)

    # output
    with open(os.path.join(outputDir, f"statistics_{idx}.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")

        for key in stats:
            outputValue(w, metric, key, stats)


def calculateStats(data):

    # get non-zero data following the (+1 data / -1 result) approach
    nonZeroData = [value + 1 for value in data]

    stats = dict(
        count=len(data),
        min=min(data),
        max=max(data),
        mean=mean(data),
        # not used because we lack rules on how to deal with negatives
        # geometric_mean=geometric_mean(nonZeroData) - 1,
        # harmonic_mean=harmonic_mean(data),
        median=median(data),
        median_low=median_low(data),
        median_high=median_high(data),
        median_grouped=median_grouped(data),
        mode=mode(data),
        pstdev=pstdev(data),
        pvariance=pvariance(data),
        stdev=stdev(data) if len(data) > 1 else None,
        variance=variance(data) if len(data) > 1 else None,
    )

    return stats


def outputValue(w, metric: str, name: str, dict: dict):
    value = dict[name]
    name = "{0}_{1}".format(metric, name)
    w.writerow([name, value])

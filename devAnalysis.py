import os
import csv

from configuration import Configuration


def devAnalysis(authorInfoDict: set, batchIdx: int, devs: set, config: Configuration):

    # select experienced developers
    experiencedDevs = [
        login
        for login, author in authorInfoDict.items()
        if author["experienced"] == True
    ]

    # filter by developers present in list of aliased developers by commit
    numberActiveExperiencedDevs = len(devs.intersection(set(experiencedDevs)))

    print("Writing active experienced developer analysis results")
    with open(
        os.path.join(config.resultsPath, f"results_{batchIdx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberActiveExperiencedDevs", numberActiveExperiencedDevs])
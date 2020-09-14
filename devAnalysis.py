import os
import csv

def devAnalysis(authorInfoDict: set, batchIdx: int, devs: set, outputDir: str):
    
    # select experienced developers
    experiencedDevs = [login for login, author in authorInfoDict.items() if author['experienced'] == True]

    # filter by developers present in list of aliased developers by commit
    numberActiveExperiencedDevs = len(devs.intersection(set(experiencedDevs)))
    
    print("Writing active experienced developer analysis results")
    with open(os.path.join(outputDir, f"project_{batchIdx}.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberActiveExperiencedDevs", numberActiveExperiencedDevs])
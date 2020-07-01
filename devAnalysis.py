import os
import csv

def devAnalysis(authorInfoDict: set, issueOrPrDevs: set, outputDir: str):
    
    # select experienced developers
    experiencedDevs = [login for login, author in authorInfoDict.items() if author['experienced'] == True]

    # filter by developers present in list of aliased developers by commit
    numberActiveExperiencedDevs = len(issueOrPrDevs.intersection(set(experiencedDevs)))
    
    print("Writing active experienced developer analysis results")
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberActiveExperiencedDevs", numberActiveExperiencedDevs])

def splitRepoName(repoShortName: str):
    split = repoShortName.split("/")
    return split[0], split[1]

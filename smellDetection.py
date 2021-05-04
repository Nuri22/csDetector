import csv
import os
import warnings

from joblib import load
from configuration import Configuration

warnings.filterwarnings("ignore") 
def smellDetection(config: Configuration, batchIdx: int):

    # prepare results holder for easy mapping
    results = {}

    # open finalized results for reading
    project_csv_path = os.path.join(config.resultsPath, f"results_{batchIdx}.csv")
    with open(project_csv_path, newline="") as csvfile:
        rows = csv.reader(csvfile, delimiter=",")

        # parse into a dictionary
        for row in rows:
            results[row[0]] = row[1]

    # map results to a list suitable for model classification
    metrics = buildMetricsList(results)

    # load all models
    smells = ["OSE", "BCE", "PDE", "SV", "OS", "SD", "RS", "TF", "UI", "TC"]
    all_models = {}
    for smell in smells:
        modelPath = os.path.abspath("./models/{}.joblib".format(smell))
        all_models[smell] = load(modelPath)

    # detect smells
    rawSmells = {smell: all_models[smell].predict(metrics) for smell in all_models}
    detectedSmells = [smell for smell in smells if rawSmells[smell][0] == 1]

    # add last commit date as first output param
    detectedSmells.insert(0, results["LastCommitDate"])

    # display results
    print("Detected smells:")
    print(detectedSmells)


def buildMetricsList(results: dict):

    # declare names to extract from the results file in the right order
    names = [
        "AuthorCount",
        "DaysActive",
        "CommitCount",
        "AuthorCommitCount_stdev",
        "commitCentrality_NumberHighCentralityAuthors",
        "commitCentrality_PercentageHighCentralityAuthors",
        "SponsoredAuthorCount",
        "PercentageSponsoredAuthors",
        "NumberPRs",
        "PRParticipantsCount_stdev",
        "PRParticipantsCount_mean",
        "NumberIssues",
        "IssueParticipantCount_stdev",
        "IssueCountPositiveComments_mean",
        "commitCentrality_Centrality_count",
        "commitCentrality_Centrality_stdev",
        "commitCentrality_Betweenness_count",
        "commitCentrality_Closeness_count",
        "commitCentrality_Density",
        "commitCentrality_CommunityAuthorCount_count",
        "commitCentrality_CommunityAuthorItemCount_mean",
        "commitCentrality_CommunityAuthorItemCount_stdev",
        "commitCentrality_CommunityAuthorCount_mean",
        "commitCentrality_CommunityAuthorCount_stdev",
        "TimezoneCount",
        "TimezoneCommitCount_mean",
        "TimezoneCommitCount_stdev",
        "TimezoneAuthorCount_mean",
        "TimezoneAuthorCount_stdev",
        "NumberReleases",
        "ReleaseCommitCount_mean",
        "ReleaseCommitCount_stdev",
        "FN",
        "PRDuration_mean",
        "IssueDuration_mean",
        "BusFactorNumber",
        "commitCentrality_TFN",
        "commitCentrality_TFC",
        "PRCommentsCount_mean",
        "PRCommitsCount_mean",
        "NumberIssueComments",
        "IssueCommentsCount_mean",
        "IssueCommentsCount_stdev",
        "PRCommentsToxicityPercentage",
        "IssueCommentsToxicityPercentage",
        "RPCPR",
        "RPCIssue",
        "IssueCountNegativeComments_mean",
        "PRCountNegativeComments_mean",
        "ACCL"
    ]

    # build key/value list
    metrics = []
    for name in names:

        # default value if key isn't present or the value is blank
        result = results.get(name, 0)
        if not result:

            print(f"No value for '{name}' during smell detection, defaulting to 0")
            result = 0

        metrics.append(result)

    # return as a 2D array
    return [metrics]

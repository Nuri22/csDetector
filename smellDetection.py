import csv
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, libsvm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, auc, f1_score, r2_score, mean_squared_error
from sklearn.metrics import precision_score, precision_recall_curve, cohen_kappa_score, roc_curve
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import warnings
import os
import pickle
from joblib import dump, load

from configuration import Configuration


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
    smells = ["OSE", "BCE", "PDE", "SV", "OS", "SD", "RS", "TF", "UI", "UO", "VS"]
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
    metrics = []

    metrics.append(results.get("CommitCount", 0))
    metrics.append(results.get("DaysActive", 0))
    metrics.append(results.get("AuthorCount", 0))
    metrics.append(results.get("SponsoredAuthorCount", 0))
    metrics.append(results.get("PercentageSponsoredAuthors", 0))
    metrics.append(results.get("TimezoneCount", 0))
    metrics.append(results.get("AuthorActiveDays_mean", 0))
    metrics.append(results.get("AuthorActiveDays_stdev", 0))
    metrics.append(results.get("AuthorCommitCount_stdev", 0))
    metrics.append(results.get("TimezoneAuthorCount_stdev", 0))
    metrics.append(results.get("TimezoneCommitCount_stdev", 0))
    metrics.append(results.get("CommitMessageSentiment_mean", 0))
    metrics.append(results.get("CommitMessageSentiment_stdev", 0))
    metrics.append(results.get("CommitMessageSentimentsPositive_count", 0))
    metrics.append(results.get("CommitMessageSentimentsPositive_mean", 0))
    metrics.append(results.get("CommitMessageSentimentsPositive_stdev", 0))
    metrics.append(results.get("CommitMessageSentimentsNegative_count", 0))
    metrics.append(results.get("CommitMessageSentimentsNegative_mean", 0))
    metrics.append(results.get("Tag Count", 0))
    metrics.append(results.get("TagCommitCount_stdev", 0))
    metrics.append(results.get("commitCentrality_Density", 0))
    metrics.append(results.get("commitCentrality_Community Count", 0))
    metrics.append(results.get("commitCentrality_NumberHighCentralityAuthors", 0))
    metrics.append(results.get("commitCentrality_PercentageHighCentralityAuthors", 0))
    metrics.append(results.get("commitCentrality_Closeness_stdev", 0))
    metrics.append(results.get("commitCentrality_Betweenness_stdev", 0))
    metrics.append(results.get("commitCentrality_Centrality_stdev", 0))
    metrics.append(results.get("commitCentrality_CommunityAuthorCount_stdev", 0))
    metrics.append(results.get("commitCentrality_CommunityAuthorItemCount_stdev", 0))
    metrics.append(results.get("NumberReleases", 0))
    metrics.append(results.get("ReleaseAuthorCount_stdev", 0))
    metrics.append(results.get("ReleaseCommitCount_stdev", 0))
    metrics.append(results.get("NumberPRs", 0))
    metrics.append(results.get("NumberPRComments", 0))
    metrics.append(results.get("PRCommentsCount_mean", 0))
    metrics.append(results.get("PRCommitsCount_stdev", 0))
    metrics.append(results.get("PRCommentSentiments_stdev", 0))
    metrics.append(results.get("PRParticipantsCount_mean", 0))
    metrics.append(results.get("PRParticipantsCount_stdev", 0))
    metrics.append(results.get("PRCountPositiveComments_count", 0))
    metrics.append(results.get("PRCountPositiveComments_mean", 0))
    metrics.append(results.get("PRCountNegativeComments_count", 0))
    metrics.append(results.get("PRCountNegativeComments_mean", 0))
    metrics.append(results.get("NumberIssues", 0))
    metrics.append(results.get("NumberIssueComments", 0))
    metrics.append(results.get("NumberIssueCommentsPositive", 0))
    metrics.append(results.get("IssueCommentsCount_mean", 0))
    metrics.append(results.get("IssueCommentsCount_stdev", 0))
    metrics.append(results.get("IssueCommentSentiments_stdev", 0))
    metrics.append(results.get("IssueParticipantCount_mean", 0))
    metrics.append(results.get("IssueParticipantCount_stdev", 0))
    metrics.append(results.get("IssueCountPositiveComments_mean", 0))
    metrics.append(results.get("IssueCountNegativeComments_count", 0))
    metrics.append(results.get("IssueCountNegativeComments_mean", 0))
    metrics.append(results.get("issuesAndPRsCentrality_Density", 0))
    metrics.append(results.get("issuesAndPRsCentrality_Community Count", 0))
    metrics.append(results.get("issuesAndPRsCentrality_NumberHighCentralityAuthors", 0))
    metrics.append(results.get("issuesAndPRsCentrality_PercentageHighCentralityAuthors", 0))
    metrics.append(results.get("issuesAndPRsCentrality_Closeness_stdev", 0))
    metrics.append(results.get("issuesAndPRsCentrality_Betweenness_stdev", 0))
    metrics.append(results.get("issuesAndPRsCentrality_Centrality_stdev", 0))
    metrics.append(results.get("issuesAndPRsCentrality_CommunityAuthorCount_stdev", 0))
    metrics.append(results.get("issuesAndPRsCentrality_CommunityAuthorItemCount_stdev", 0))

    return [metrics]
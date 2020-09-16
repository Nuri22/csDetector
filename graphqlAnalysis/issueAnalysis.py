import os
import csv
import statsAnalysis as stats
import sentistrength
import graphqlAnalysis.graphqlAnalysisHelper as gql
from functools import reduce
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
from typing import List
from datetime import datetime
from configuration import Configuration
import threading


def issueAnalysis(
    config: Configuration,
    pat: str,
    senti: sentistrength.PySentiStr,
    delta: relativedelta,
    batchDates: List[datetime],
):

    # split repo by owner and name
    owner, name = gql.splitRepoName(config.repositoryShortname)

    print("Querying issue comments")
    batches = issueRequest(pat, owner, name, delta, batchDates)

    participantBatches = list()

    for batchIdx, batch in enumerate(batches):
        print(f"Analyzing issue batch #{batchIdx}")

        # extract data from batch
        issueCount = len(batch)
        participants = set(p for pr in batch for p in pr["participants"])
        participantBatches.append(participants)

        allComments = list()
        issuePositiveComments = list()
        issueNegativeComments = list()

        print(f"    Sentiments per issue", end="")

        semaphore = threading.Semaphore(255)
        threads = []
        for pr in batch:

            prComments = pr["comments"]

            if len(prComments) == 0:
                issuePositiveComments.append(0)
                issueNegativeComments.append(0)
                continue

            allComments.extend(prComments)

            thread = threading.Thread(
                target=analyzeSentiments,
                args=(
                    senti,
                    prComments,
                    issuePositiveComments,
                    issueNegativeComments,
                    semaphore,
                ),
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        print("")

        print("    All sentiments")

        # analyze comment issue sentiment
        issueCommentSentiments = []
        issueCommentSentimentsPositive = 0
        issueCommentSentimentsNegative = 0

        if len(allComments) > 0:
            issueCommentSentiments = senti.getSentiment(allComments)
            issueCommentSentimentsPositive = sum(
                1 for _ in filter(lambda value: value >= 1, issueCommentSentiments)
            )
            issueCommentSentimentsNegative = sum(
                1 for _ in filter(lambda value: value <= -1, issueCommentSentiments)
            )

        print("Writing GraphQL analysis results")
        with open(
            os.path.join(config.analysisOutputPath, f"project_{batchIdx}.csv"),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["NumberIssues", len(batch)])
            w.writerow(["NumberIssueComments", len(allComments)])
            w.writerow(["NumberIssueCommentsPositive", issueCommentSentimentsPositive])
            w.writerow(["NumberIssueCommentsNegative", issueCommentSentimentsNegative])

        with open(
            os.path.join(
                config.analysisOutputPath, f"issueCommentsCount_{batchIdx}.csv"
            ),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["Issue Number", "Comment Count"])
            for issue in batch:
                w.writerow([issue["number"], len(issue["comments"])])

        with open(
            os.path.join(
                config.analysisOutputPath, f"issueParticipantCount_{batchIdx}.csv"
            ),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["Issue Number", "Developer Count"])
            for issue in batch:
                w.writerow([issue["number"], len(issue["participants"])])

        # output statistics
        stats.outputStatistics(
            batchIdx,
            [len(issue["comments"]) for issue in batch],
            "IssueCommentsCount",
            config.analysisOutputPath,
        )

        stats.outputStatistics(
            batchIdx,
            issueCommentSentiments,
            "IssueCommentSentiments",
            config.analysisOutputPath,
        )

        stats.outputStatistics(
            batchIdx,
            [len(issue["participants"]) for issue in batch],
            "IssueParticipantCount",
            config.analysisOutputPath,
        )

        stats.outputStatistics(
            batchIdx,
            issuePositiveComments,
            "IssueCountPositiveComments",
            config.analysisOutputPath,
        )

        stats.outputStatistics(
            batchIdx,
            issueNegativeComments,
            "IssueCountNegativeComments",
            config.analysisOutputPath,
        )

    return participantBatches


def analyzeSentiments(
    senti, prComments, prPositiveComments, prNegativeComments, semaphore
):
    with semaphore:
        commentSentiments = senti.getSentiment(prComments)
        commentSentimentsPositive = sum(
            1 for _ in filter(lambda value: value >= 1, commentSentiments)
        )
        commentSentimentsNegative = sum(
            1 for _ in filter(lambda value: value <= -1, commentSentiments)
        )

        lock = threading.Lock()
        with lock:
            prPositiveComments.append(commentSentimentsPositive)
            prNegativeComments.append(commentSentimentsNegative)
            print(f".", end="")


def issueRequest(
    pat: str, owner: str, name: str, delta: relativedelta, batchDates: List[datetime]
):

    # prepare batches
    batches = []
    batch = None
    batchStartDate = None
    batchEndDate = None

    cursor = None
    while True:

        # get page of PRs
        query = buildIssueRequestQuery(owner, name, cursor)
        result = gql.runGraphqlRequest(pat, query)
        print("...")

        # extract nodes
        nodes = result["repository"]["issues"]["nodes"]

        # analyse
        for node in nodes:

            createdAt = isoparse(node["createdAt"])

            if batchEndDate == None or createdAt > batchEndDate:

                if batch != None:
                    batches.append(batch)

                batchStartDate = batchDates[len(batches)]
                batchEndDate = batchStartDate + delta

                batch = []

            issue = {
                "number": node["number"],
                "createdAt": createdAt,
                "comments": list(c["bodyText"] for c in node["comments"]["nodes"]),
                "participants": set(),
            }

            # participants
            participantCount = 0
            for user in node["participants"]["nodes"]:
                if gql.tryAddLogin(user, issue["participants"]):
                    participantCount += 1

            batch.append(issue)

        # check for next page
        pageInfo = result["repository"]["issues"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]

    if batch != None:
        batches.append(batch)

    return batches


def buildIssueRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            issues(first: 100{2}) {{
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    number
                    createdAt
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    comments(first: 100) {{
                        nodes {{
                            bodyText
                        }}
                    }}
                }}
            }}
        }}
    }}""".format(
        owner, name, gql.buildNextPageQuery(cursor)
    )

import os
import csv
import statsAnalysis as stats
import sentistrength
import graphqlAnalysis.graphqlAnalysisHelper as gql
import centralityAnalysis as centrality
from functools import reduce
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
from typing import List
from datetime import datetime
from configuration import Configuration
import threading
from collections import Counter


def issueAnalysis(
    config: Configuration,
    senti: sentistrength.PySentiStr,
    delta: relativedelta,
    batchDates: List[datetime],
):

    print("Querying issue comments")
    batches = issueRequest(config.pat, config.repositoryOwner, config.repositoryName, delta, batchDates)

    batchParticipants = list()

    for batchIdx, batch in enumerate(batches):
        print(f"Analyzing issue batch #{batchIdx}")

        # extract data from batch
        issueCount = len(batch)
        issueParticipants = list(
            issue["participants"] for issue in batch if len(issue["participants"]) > 0
        )
        batchParticipants.append(issueParticipants)

        allComments = list()
        issuePositiveComments = list()
        issueNegativeComments = list()

        print(f"    Sentiments per issue", end="")

        semaphore = threading.Semaphore(15)
        threads = []
        for issue in batch:

            prComments = list(
                comment for comment in issue["comments"] if comment and comment.strip()
            )

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

        # centrality.buildGraphQlNetwork(
        #     batchIdx, issueParticipants, "Issues", config.metricsPath
        # )

        print("Writing GraphQL analysis results")
        with open(
            os.path.join(config.resultsPath, f"results_{batchIdx}.csv"),
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
                config.metricsPath, f"issueCommentsCount_{batchIdx}.csv"
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
                config.metricsPath, f"issueParticipantCount_{batchIdx}.csv"
            ),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["Issue Number", "Developer Count"])
            for issue in batch:
                w.writerow([issue["number"], len(set(issue["participants"]))])

        # output statistics
        stats.outputStatistics(
            batchIdx,
            [len(issue["comments"]) for issue in batch],
            "IssueCommentsCount",
            config.metricsPath,
        )

        stats.outputStatistics(
            batchIdx,
            issueCommentSentiments,
            "IssueCommentSentiments",
            config.metricsPath,
        )

        stats.outputStatistics(
            batchIdx,
            [len(set(issue["participants"])) for issue in batch],
            "IssueParticipantCount",
            config.metricsPath,
        )

        stats.outputStatistics(
            batchIdx,
            issuePositiveComments,
            "IssueCountPositiveComments",
            config.metricsPath,
        )

        stats.outputStatistics(
            batchIdx,
            issueNegativeComments,
            "IssueCountNegativeComments",
            config.metricsPath,
        )

    return batchParticipants


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

            if batchEndDate == None or (
                createdAt > batchEndDate and len(batches) < len(batchDates) - 1
            ):
                if batch != None:
                    batches.append(batch)

                batchStartDate = batchDates[len(batches)]
                batchEndDate = batchStartDate + delta

                batch = []

            issue = {
                "number": node["number"],
                "createdAt": createdAt,
                "comments": list(c["bodyText"] for c in node["comments"]["nodes"]),
                "participants": list(),
            }

            # participants
            for user in node["participants"]["nodes"]:
                gql.addLogin(user, issue["participants"])

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

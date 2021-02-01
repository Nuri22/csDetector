import math
import os
import csv
import sys
from perspectiveAnalysis import getToxicityPercentage
import statsAnalysis as stats
import sentistrength
import graphqlAnalysis.graphqlAnalysisHelper as gql
import centralityAnalysis as centrality
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
from typing import List
from datetime import date, datetime, timezone
from configuration import Configuration
import itertools
import threading
from collections import Counter


def prAnalysis(
    config: Configuration,
    senti: sentistrength.PySentiStr,
    delta: relativedelta,
    batchDates: List[datetime],
):

    print("Querying PRs")
    batches = prRequest(
        config.pat, config.repositoryOwner, config.repositoryName, delta, batchDates
    )

    batchParticipants = list()
    batchComments = list()

    for batchIdx, batch in enumerate(batches):
        print(f"Analyzing PR batch #{batchIdx}")

        # extract data from batch
        prCount = len(batch)
        participants = list(
            pr["participants"] for pr in batch if len(pr["participants"]) > 0
        )
        batchParticipants.append(participants)

        allComments = list()
        prPositiveComments = list()
        prNegativeComments = list()
        generallyNegative = list()

        print(f"    Sentiments per PR", end="")

        semaphore = threading.Semaphore(15)
        threads = []
        for pr in batch:

            comments = list(
                comment for comment in pr["comments"] if comment and comment.strip()
            )

            # split comments that are longer than 20KB
            splitComments = []
            for comment in comments:

                # calc number of chunks
                byteChunks = math.ceil(sys.getsizeof(comment) / (20 * 1024))
                if byteChunks > 1:

                    # calc desired max length of each chunk
                    chunkLength = math.floor(len(comment) / byteChunks)

                    # divide comment into chunks
                    chunks = [
                        comment[i * chunkLength : i * chunkLength + chunkLength]
                        for i in range(0, byteChunks)
                    ]

                    # save chunks
                    splitComments.extend(chunks)

                else:
                    # append comment as-is
                    splitComments.append(comment)

            # re-assign comments after chunking
            comments = splitComments

            if len(comments) == 0:
                prPositiveComments.append(0)
                prNegativeComments.append(0)
                continue

            allComments.extend(comments)

            thread = threading.Thread(
                target=analyzeSentiments,
                args=(
                    senti,
                    comments,
                    prPositiveComments,
                    prNegativeComments,
                    generallyNegative,
                    semaphore,
                ),
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        print("")

        # save comments
        batchComments.append(allComments)

        # get comment length stats
        commentLengths = [len(c) for c in allComments]

        generallyNegativeRatio = len(generallyNegative) / prCount

        # get pr duration stats
        durations = [(pr["closedAt"] - pr["createdAt"]).days for pr in batch]

        print("    All sentiments")

        commentSentiments = []
        commentSentimentsPositive = 0
        commentSentimentsNegative = 0

        if len(allComments) > 0:
            commentSentiments = senti.getSentiment(allComments)
            commentSentimentsPositive = sum(
                1 for _ in filter(lambda value: value >= 1, commentSentiments)
            )
            commentSentimentsNegative = sum(
                1 for _ in filter(lambda value: value <= -1, commentSentiments)
            )

        toxicityPercentage = getToxicityPercentage(config, allComments)

        centrality.buildGraphQlNetwork(batchIdx, participants, "PRs", config)

        print("    Writing results")
        with open(
            os.path.join(config.resultsPath, f"results_{batchIdx}.csv"),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["NumberPRs", prCount])
            w.writerow(["NumberPRComments", len(allComments)])
            w.writerow(["PRCommentsPositive", commentSentimentsPositive])
            w.writerow(["PRCommentsNegative", commentSentimentsNegative])
            w.writerow(["PRCommentsNegativeRatio", generallyNegativeRatio])
            w.writerow(["PRCommentsToxicityPercentage", toxicityPercentage])

        with open(
            os.path.join(config.metricsPath, f"PRCommits_{batchIdx}.csv"),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["PR Number", "Commit Count"])
            for pr in batch:
                w.writerow([pr["number"], pr["commitCount"]])

        with open(
            os.path.join(config.metricsPath, f"PRParticipants_{batchIdx}.csv"),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["PR Number", "Developer Count"])
            for pr in batch:
                w.writerow([pr["number"], len(set(pr["participants"]))])

        # output statistics
        stats.outputStatistics(
            batchIdx,
            commentLengths,
            "PRCommentsLength",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            durations,
            "PRDuration",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            [len(pr["comments"]) for pr in batch],
            "PRCommentsCount",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            [pr["commitCount"] for pr in batch],
            "PRCommitsCount",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            commentSentiments,
            "PRCommentSentiments",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            [len(set(pr["participants"])) for pr in batch],
            "PRParticipantsCount",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            prPositiveComments,
            "PRCountPositiveComments",
            config.resultsPath,
        )

        stats.outputStatistics(
            batchIdx,
            prNegativeComments,
            "PRCountNegativeComments",
            config.resultsPath,
        )

    return batchParticipants, batchComments


def analyzeSentiments(
    senti, comments, positiveComments, negativeComments, generallyNegative, semaphore
):
    with semaphore:
        commentSentiments = (
            senti.getSentiment(comments, score="scale")
            if len(comments) > 1
            else senti.getSentiment(comments[0])
        )

        commentSentimentsPositive = sum(
            1 for _ in filter(lambda value: value >= 1, commentSentiments)
        )
        commentSentimentsNegative = sum(
            1 for _ in filter(lambda value: value <= -1, commentSentiments)
        )

        lock = threading.Lock()
        with lock:
            positiveComments.append(commentSentimentsPositive)
            negativeComments.append(commentSentimentsNegative)

            if commentSentimentsNegative / len(comments) > 0.5:
                generallyNegative.append(True)

            print(f".", end="")


def prRequest(
    pat: str, owner: str, name: str, delta: relativedelta, batchDates: List[datetime]
):
    query = buildPrRequestQuery(owner, name, None)

    # prepare batches
    batches = []
    batch = None
    batchStartDate = None
    batchEndDate = None

    while True:

        # get page
        result = gql.runGraphqlRequest(pat, query)
        print("...")

        # extract nodes
        nodes = result["repository"]["pullRequests"]["nodes"]

        # add results
        for node in nodes:

            createdAt = isoparse(node["createdAt"])
            closedAt = (
                datetime.now(timezone.utc)
                if node["closedAt"] is None
                else isoparse(node["closedAt"])
            )

            if batchEndDate == None or (
                createdAt > batchEndDate and len(batches) < len(batchDates) - 1
            ):

                if batch != None:
                    batches.append(batch)

                batchStartDate = batchDates[len(batches)]
                batchEndDate = batchStartDate + delta

                batch = []

            pr = {
                "number": node["number"],
                "createdAt": createdAt,
                "closedAt": closedAt,
                "comments": list(c["bodyText"] for c in node["comments"]["nodes"]),
                "commitCount": node["commits"]["totalCount"],
                "participants": list(),
            }

            # participants
            for user in node["participants"]["nodes"]:
                gql.addLogin(user, pr["participants"])

            batch.append(pr)

        # check for next page
        pageInfo = result["repository"]["pullRequests"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildPrRequestQuery(owner, name, cursor)

    if batch != None:
        batches.append(batch)

    return batches


def buildPrRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            pullRequests(first:100{2}) {{
                pageInfo {{
                    endCursor
                    hasNextPage
                }}
                nodes {{
                    number
                    createdAt
                    closedAt
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    commits {{
                        totalCount
                    }}
                    comments(first: 100) {{
                        nodes {{
                            bodyText
                        }}
                    }}
                }}
            }}
        }}
    }}
    """.format(
        owner, name, gql.buildNextPageQuery(cursor)
    )

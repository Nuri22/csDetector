import os
import csv
import statsAnalysis as stats
import sentistrength
import graphqlAnalysis.graphqlAnalysisHelper as gql

def prAnalysis(
    pat: str, senti: sentistrength.PySentiStr, repoShortName: str, outputDir: str
):
    # split repo by owner and name
    owner, name = gql.splitRepoName(repoShortName)

    print("Querying PRs")
    (prCount, prCommitCounts, prComments, participants, participantCount) = prRequest(
        pat, owner, name
    )

    # analyze sentiments
    commentSentiments = senti.getSentiment(prComments)
	commentSentimentsPositive = sum(
        1 for _ in filter(lambda value: value >= 1, commentSentiments)
    )
    commentSentimentsNegative = sum(
        1 for _ in filter(lambda value: value <= -1, commentSentiments)
    )

    print("Writing results")
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberPRs", prCount])
        w.writerow(["NumberPRComments", prCount])
		w.writerow(["PRCommentsPositive", commentSentimentsPositive])
        w.writerow(["PRCommentsNegative", commentSentimentsNegative])
       

    with open(os.path.join(outputDir, "PRCommits.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["PR Number", "Commit Count"])
        for key, value in prCommitCounts.items():
            w.writerow([key, value])

    with open(os.path.join(outputDir, "PRParticipants.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["PR Number", "Developer Count"])
        for key, value in participantCount.items():
            w.writerow([key, value])

    stats.outputStatistics(
        [value for key, value in prCommitCounts.items()], "PRCommitsCount", outputDir
    )

    stats.outputStatistics(
        commentSentiments, "PRCommentSentiments", outputDir,
    )

    stats.outputStatistics(
        [value for key, value in participantCount.items()],
        "PRParticipantsCount",
        outputDir,
    )

    return participants


def prRequest(pat: str, owner: str, name: str):
    query = buildPrRequestQuery(owner, name, None)

    prCount = 0
    prCommitCounts = {}
    prComments = []
    prCommentCount = 0
    participants = set()
    participantCount = dict()

    while True:

        # get page
        result = gql.runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["pullRequests"]["nodes"]

        if prCount == 0:
            prCount = result["repository"]["pullRequests"]["totalCount"]

        # add results
        for node in nodes:
            prCommentCount += node["comments"]["totalCount"]
            commitCount = node["commits"]["totalCount"]
            prCommitCounts[node["number"]] = commitCount

            prComments.extend(c["bodyText"] for c in node["comments"]["nodes"])

            prParticipantCount = 0

            # author
            if gql.tryAddLogin(node["author"], participants):
                prParticipantCount += 1

            # editor
            if gql.tryAddLogin(node["editor"], participants):
                prParticipantCount += 1

            # assignees
            for user in node["assignees"]["nodes"]:
                if gql.tryAddLogin(user, participants):
                    prParticipantCount += 1

            # participants
            for user in node["participants"]["nodes"]:
                if gql.tryAddLogin(user, participants):
                    prParticipantCount += 1

            participantCount[node["number"]] = prParticipantCount

        # check for next page
        pageInfo = result["repository"]["pullRequests"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildPrRequestQuery(owner, name, cursor)

    return (prCount, prCommitCounts, prComments, participants, participantCount)


def buildPrRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            pullRequests(first:100{2}) {{
                totalCount
                pageInfo {{
                    endCursor
                    hasNextPage
                }}
                nodes {{
                    number
                    author {{
                        ... on User {{
                            login
                        }}
                    }}
                    editor {{
                        ... on User {{
                            login
                        }}
                    }}
                    assignees(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    commits {{
                        totalCount
                    }}
                    comments(first: 100) {{
                        totalCount
                        nodes {{
                            bodyText
                        }}
                        totalCount
                    }}
                }}
            }}
        }}
    }}
    """.format(
        owner, name, gql.buildNextPageQuery(cursor)
    )

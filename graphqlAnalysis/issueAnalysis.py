import os
import csv
import statsAnalysis as stats
import sentistrength
import graphqlAnalysis.graphqlAnalysisHelper as gql
from functools import reduce


def issueAnalysis(
    pat: str, senti: sentistrength.PySentiStr, repoShortName: str, outputDir: str
):

    # split repo by owner and name
    owner, name = gql.splitRepoName(repoShortName)

    print("Querying issue comments")
    (issueCount, issues, participants) = issueRequest(pat, owner, name)

    commentCount = 0
    issueComments = []
    for key, value in issues.items():
        commentCount += value["commentCount"]
        comments = value["comments"]

        # remove empty comments from list
        while "" in comments:
            comments.remove("")

        # add to main list
        issueComments.extend(comments)

    # analyze comment issue sentiment
    issueCommentSentiments = []
    issueCommentSentimentsPositive = 0
    issueCommentSentimentsNegative = 0

    if len(issueComments) > 0:
        issueCommentSentiments = senti.getSentiment(issueComments)
        issueCommentSentimentsPositive = sum(
            1 for _ in filter(lambda value: value >= 1, issueCommentSentiments)
        )
        issueCommentSentimentsNegative = sum(
            1 for _ in filter(lambda value: value <= -1, issueCommentSentiments)
        )

    print("Writing GraphQL analysis results")
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberIssues", issueCount])
        w.writerow(["NumberIssueComments", commentCount])
        w.writerow(["NumberIssueCommentsPositive", issueCommentSentimentsPositive])
        w.writerow(["NumberIssueCommentsNegative", issueCommentSentimentsNegative])

    with open(os.path.join(outputDir, "issueCommentsCount.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Issue Number", "Commit Count"])
        for key, value in issues.items():
            w.writerow([key, value["commentCount"]])

    with open(
        os.path.join(outputDir, "issueParticipantCount.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Issue Number", "Developer Count"])
        for key, value in issues.items():
            w.writerow([key, value["participantCount"]])

    # output statistics
    stats.outputStatistics(
        [value["commentCount"] for key, value in issues.items()],
        "IssueCommentsCount",
        outputDir,
    )

    stats.outputStatistics(
        issueCommentSentiments, "IssueCommentSentiments", outputDir,
    )

    stats.outputStatistics(
        [value["participantCount"] for key, value in issues.items()],
        "IssueParticipantCount",
        outputDir,
    )

    return participants


def issueRequest(pat: str, owner: str, name: str):

    issueCount = 0
    issues = dict()
    participants = set()

    cursor = None
    while True:

        # get page of PRs
        query = buildIssueRequestQuery(owner, name, cursor)
        result = gql.runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["issues"]["nodes"]

        if issueCount == 0:
            issueCount = result["repository"]["issues"]["totalCount"]

        # analyse
        for node in nodes:
            commentCount = node["comments"]["totalCount"]
            comments = [comment["bodyText"] for comment in node["comments"]["nodes"]]

            # participants
            participantCount = 0
            for user in node["participants"]["nodes"]:
                if gql.tryAddLogin(user, participants):
                    participantCount += 1

            issues[node["number"]] = dict(
                commentCount=commentCount,
                comments=comments,
                participantCount=participantCount,
            )

        # check for next page
        pageInfo = result["repository"]["issues"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]

    return (issueCount, issues, participants)


def buildIssueRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            issues(first: 100{2}) {{
                totalCount
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    number
                    participants(first: 100) {{
                        nodes {{
                            login
                        }}
                    }}
                    comments(first: 100) {{
                        totalCount
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

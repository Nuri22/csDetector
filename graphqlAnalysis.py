import graphqlRequests as gql
import os
import csv
from statsAnalysis import outputStatistics


def graphqlAnalysis(pat: str, repoShortName: str, outputDir: str):

    # split repo by owner and name
    owner, name = splitRepoName(repoShortName)

    print("Querying number of issues")
    issueCount = gql.countIssuesPerRepository(pat, owner, name)

    print("Querying number of PRs")
    prCount = gql.countPullRequestsPerRepository(pat, owner, name)

    print("Querying number of commits per PR")
    prCommitCount = gql.countCommitsPerPullRequest(pat, owner, name)

    print("Querying issue participants")
    issueParticipants, issueParticipantCount = gql.getIssueParticipants(
        pat, owner, name
    )

    print("Querying PR participants")
    prParticipants, prParticipantCount = gql.getPullRequestParticipants(
        pat, owner, name
    )

    # join lists and clean memory
    participants = issueParticipants.union(prParticipants)
    del issueParticipants
    del prParticipants

    print("Querying number of comments per issue")
    issueCommentCount = gql.getIssueComments(pat, owner, name)

    print("Writing GraphQL analysis results")
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberIssues", issueCount])
        w.writerow(["NumberPRs", prCount])

    with open(os.path.join(outputDir, "numberCommitsPR.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["PR Number", "Commit Count"])
        for key, value in prCommitCount.items():
            w.writerow([key, value])

    with open(
        os.path.join(outputDir, "numberDevelopersIssue.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Issue Number", "Developer Count"])
        for key, value in issueParticipantCount.items():
            w.writerow([key, value])

    with open(os.path.join(outputDir, "numberDevelopersPR.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["PR Number", "Developer Count"])
        for key, value in prParticipantCount.items():
            w.writerow([key, value])

    with open(os.path.join(outputDir, "numberCommentsIssue.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["PR Number", "Commit Count"])
        for key, value in issueCommentCount.items():
            w.writerow([key, value])

    # output statistics
    outputStatistics(
        [value for key, value in prCommitCount.items()], "CommitsPRCount", outputDir
    )

    outputStatistics(
        [value for key, value in issueParticipantCount.items()],
        "DevelopersIssueCount",
        outputDir,
    )

    outputStatistics(
        [value for key, value in prParticipantCount.items()],
        "DevelopersPRCount",
        outputDir,
    )

    outputStatistics(
        [value for key, value in issueCommentCount.items()],
        "CommentsIssueCount",
        outputDir,
    )

    return participants


def splitRepoName(repoShortName: str):
    split = repoShortName.split("/")
    return split[0], split[1]

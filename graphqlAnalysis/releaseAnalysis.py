import os
import csv
import graphqlAnalysis.graphqlAnalysisHelper as gql
import dateutil
import git
import statsAnalysis as stats
from typing import List
from datetime import datetime


def releaseAnalysis(
    allCommits: List[git.Commit], pat: str, repoShortName: str, outputDir: str
):

    # split repo by owner and name
    owner, name = gql.splitRepoName(repoShortName)

    # sort commits by ascending commit date
    allCommits.sort(key=lambda c: c.committed_datetime)

    print("Querying releases")
    (releaseCount, releases) = releaseRequest(pat, owner, name)

    releaseAuthors = set()
    releaseCommitsCount = {}

    for i, release in enumerate(releases):
        releaseCommits = list()
        releaseDate = release["createdAt"]

        # try add author to set
        releaseAuthors.add(release["author"])

        if i == 0:

            # this is the first release, get all commits prior to release created date
            for commit in allCommits:
                if commit.committed_datetime < releaseDate:
                    releaseCommits.append(commit)
                else:
                    break

        else:

            # get in-between commit count
            prevReleaseDate = releases[i - 1]["createdAt"]
            for commit in allCommits:
                if (
                    commit.committed_datetime >= prevReleaseDate
                    and commit.committed_datetime < releaseDate
                ):
                    releaseCommits.append(commit)
                else:
                    break

        # remove all counted commits from list to improve iteration speed
        allCommits = allCommits[len(releaseCommits) :]

        # calculate authors per release
        commitAuthors = set(commit.author.email for commit in releaseCommits)

        # add results
        releaseCommitsCount[release["name"]] = dict(
            date=release["createdAt"],
            authorsCount=len(commitAuthors),
            commitsCount=len(releaseCommits),
        )

    # sort releases by date ascending
    releaseCommitsCount = {
        key: value
        for key, value in sorted(
            releaseCommitsCount.items(), key=lambda r: r[1]["date"]
        )
    }

    print("Writing results")
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberReleases", releaseCount])
        w.writerow(["NumberReleaseAuthors", len(releaseAuthors)])

    with open(os.path.join(outputDir, "releases.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Release", "Date", "Author Count", "Commit Count"])
        for key, value in releaseCommitsCount.items():
            w.writerow(
                [
                    key,
                    value["date"].isoformat(),
                    value["authorsCount"],
                    value["commitsCount"],
                ]
            )

    stats.outputStatistics(
        [value["authorsCount"] for key, value in releaseCommitsCount.items()],
        "ReleaseAuthorCount",
        outputDir,
    )

    stats.outputStatistics(
        [value["commitsCount"] for key, value in releaseCommitsCount.items()],
        "ReleaseCommitCount",
        outputDir,
    )


def releaseRequest(pat: str, owner: str, name: str):
    query = buildReleaseRequestQuery(owner, name, None)

    releaseCount = 0
    releases = []
    while True:

        # get page of releases
        result = gql.runGraphqlRequest(pat, query)

        # extract nodes
        nodes = result["repository"]["releases"]["nodes"]

        if releaseCount == 0:
            releaseCount = result["repository"]["releases"]["totalCount"]

        # parse
        for node in nodes:
            releases.append(
                dict(
                    name=node["name"],
                    createdAt=dateutil.parser.isoparse(node["createdAt"]),
                    author=node["author"]["login"],
                )
            )

        # check for next page
        pageInfo = result["repository"]["releases"]["pageInfo"]
        if not pageInfo["hasNextPage"]:
            break

        cursor = pageInfo["endCursor"]
        query = buildReleaseRequestQuery(owner, name, cursor)

    return (releaseCount, releases)


def buildReleaseRequestQuery(owner: str, name: str, cursor: str):
    return """{{
        repository(owner: "{0}", name: "{1}") {{
            releases(first:100{2}) {{
                totalCount
                nodes {{
                    author {{
                        login
                    }}
                    createdAt
                    name
                }}
                pageInfo {{
                    endCursor
                    hasNextPage
                }}
            }}
        }}
    }}""".format(
        owner, name, gql.buildNextPageQuery(cursor)
    )

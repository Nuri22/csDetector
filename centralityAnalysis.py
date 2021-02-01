import os
import git
from matplotlib.figure import Figure
import networkx as nx
import csv
import matplotlib.pyplot as plt

from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta
from networkx.algorithms import core
from networkx.algorithms.community import greedy_modularity_communities
from progress.bar import Bar
from collections import Counter
from utils import authorIdExtractor
from statsAnalysis import outputStatistics
from configuration import Configuration


def centralityAnalysis(
    commits: List[git.Commit],
    delta: relativedelta,
    batchDates: List[datetime],
    config: Configuration,
):
    coreDevs = list()

    # work with batched commits
    for idx, batchStartDate in enumerate(batchDates):
        batchEndDate = batchStartDate + delta

        batch = [
            commit
            for commit in commits
            if commit.committed_datetime >= batchStartDate
            and commit.committed_datetime < batchEndDate
        ]

        batchCoreDevs = processBatch(idx, batch, config)
        coreDevs.append(batchCoreDevs)

    return coreDevs


def processBatch(batchIdx: int, commits: List[git.Commit], config: Configuration):
    allRelatedAuthors = {}
    authorCommits = Counter({})

    # for all commits...
    print("Analyzing centrality")
    for commit in Bar("Processing").iter(commits):
        author = authorIdExtractor(commit.author)

        # increase author commit count
        authorCommits.update({author: 1})

        # initialize dates for related author analysis
        commitDate = datetime.fromtimestamp(commit.committed_date)
        earliestDate = commitDate + relativedelta(months=-1)
        latestDate = commitDate + relativedelta(months=+1)

        commitRelatedCommits = filter(
            lambda c: findRelatedCommits(author, earliestDate, latestDate, c), commits
        )

        commitRelatedAuthors = set(
            list(map(lambda c: authorIdExtractor(c.author), commitRelatedCommits))
        )

        # get current related authors collection and update it
        authorRelatedAuthors = allRelatedAuthors.setdefault(author, set())
        authorRelatedAuthors.update(commitRelatedAuthors)

    return prepareGraph(
        allRelatedAuthors, authorCommits, batchIdx, "commitCentrality", config
    )


def buildGraphQlNetwork(batchIdx: int, batch: list, prefix: str, config: Configuration):
    allRelatedAuthors = {}
    authorItems = Counter({})

    # for all commits...
    print("Analyzing centrality")
    for authors in batch:

        for author in authors:

            # increase author commit count
            authorItems.update({author: 1})

            # get current related authors collection and update it
            relatedAuthors = set(
                relatedAuthor
                for otherAuthors in batch
                for relatedAuthor in otherAuthors
                if author in otherAuthors and relatedAuthor != author
            )
            authorRelatedAuthors = allRelatedAuthors.setdefault(author, set())
            authorRelatedAuthors.update(relatedAuthors)

    prepareGraph(allRelatedAuthors, authorItems, batchIdx, prefix, config)


def prepareGraph(
    allRelatedAuthors: dict,
    authorItemCounts: Counter,
    batchIdx: int,
    outputPrefix: str,
    config: Configuration,
):

    # prepare graph
    print("Preparing NX graph")
    G = nx.Graph()

    for author in allRelatedAuthors:
        G.add_node(author)

        for relatedAuthor in allRelatedAuthors[author]:
            G.add_edge(author.strip(), relatedAuthor.strip())

    # analyze graph
    closeness = dict(nx.closeness_centrality(G))
    betweenness = dict(nx.betweenness_centrality(G))
    centrality = dict(nx.degree_centrality(G))
    density = nx.density(G)
    modularity = []

    try:
        for idx, community in enumerate(greedy_modularity_communities(G)):
            authorCount = len(community)
            communityCommitCount = sum(authorItemCounts[author] for author in community)
            row = [authorCount, communityCommitCount]
            modularity.append(row)
    except ZeroDivisionError:
        # not handled
        pass

    # finding high centrality authors
    highCentralityAuthors = list(
        [author for author, centrality in centrality.items() if centrality > 0.5]
    )

    numberHighCentralityAuthors = len(highCentralityAuthors)

    percentageHighCentralityAuthors = numberHighCentralityAuthors / len(
        allRelatedAuthors
    )

    print("Outputting CSVs")

    # output non-tabular results
    with open(
        os.path.join(config.resultsPath, f"results_{batchIdx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow([f"{outputPrefix}_Density", density])
        w.writerow([f"{outputPrefix}_Community Count", len(modularity)])

    # output community information
    with open(
        os.path.join(config.metricsPath, f"{outputPrefix}_community_{batchIdx}.csv"),
        "a",
        newline="",
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Community Index", "Author Count", "Item Count"])
        for idx, community in enumerate(modularity):
            w.writerow([idx + 1, community[0], community[1]])

    # combine centrality results
    combined = {}
    for key in closeness:
        single = {
            "Author": key,
            "Closeness": closeness[key],
            "Betweenness": betweenness[key],
            "Centrality": centrality[key],
        }

        combined[key] = single

    # output tabular results
    with open(
        os.path.join(config.metricsPath, f"{outputPrefix}_centrality_{batchIdx}.csv"),
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(f, ["Author", "Closeness", "Betweenness", "Centrality"])
        w.writeheader()

        for key in combined:
            w.writerow(combined[key])

    # output high centrality authors
    with open(
        os.path.join(config.resultsPath, f"results_{batchIdx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(
            [f"{outputPrefix}_NumberHighCentralityAuthors", numberHighCentralityAuthors]
        )
        w.writerow(
            [
                f"{outputPrefix}_PercentageHighCentralityAuthors",
                percentageHighCentralityAuthors,
            ]
        )

    # output statistics
    outputStatistics(
        batchIdx,
        [value for key, value in closeness.items()],
        f"{outputPrefix}_Closeness",
        config.resultsPath,
    )

    outputStatistics(
        batchIdx,
        [value for key, value in betweenness.items()],
        f"{outputPrefix}_Betweenness",
        config.resultsPath,
    )

    outputStatistics(
        batchIdx,
        [value for key, value in centrality.items()],
        f"{outputPrefix}_Centrality",
        config.resultsPath,
    )

    outputStatistics(
        batchIdx,
        [community[0] for community in modularity],
        f"{outputPrefix}_CommunityAuthorCount",
        config.resultsPath,
    )

    outputStatistics(
        batchIdx,
        [community[1] for community in modularity],
        f"{outputPrefix}_CommunityAuthorItemCount",
        config.resultsPath,
    )

    # output graph
    print("Outputting graph")
    graphFigure = plt.figure(5, figsize=(30, 30))
    nx.draw(
        G,
        with_labels=True,
        node_color="orange",
        node_size=4000,
        edge_color="black",
        linewidths=2,
        font_size=20,
    )
    graphFigure.savefig(
        os.path.join(config.resultsPath, f"{outputPrefix}_{batchIdx}.pdf")
    )

    nx.write_graphml(
        G, os.path.join(config.resultsPath, f"{outputPrefix}_{batchIdx}.xml")
    )

    return highCentralityAuthors


# helper functions
def findRelatedCommits(author, earliestDate, latestDate, commit):
    isDifferentAuthor = author != authorIdExtractor(commit.author)
    if not isDifferentAuthor:
        return False

    commitDate = datetime.fromtimestamp(commit.committed_date)
    isInRange = commitDate >= earliestDate and commitDate <= latestDate
    return isInRange

import os
import git
import networkx as nx
import csv
import matplotlib.pyplot as plt

from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta
from networkx.algorithms.community import greedy_modularity_communities
from progress.bar import Bar
from collections import Counter
from utils import authorIdExtractor
from statsAnalysis import outputStatistics


def centralityAnalysis(repo: git.Repo, commits: List[git.Commit], outputDir: str):

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

        # find authors related to this commit
        #        commitRelatedCommits = commit.iter_items(
        #                repo, 'master',
        #                after=earliestDate.strftime('%Y-%m-%d'),
        #                before=latestDate.strftime('%Y-%m-%d'))

        commitRelatedCommits = filter(
            lambda c: findRelatedCommits(author, earliestDate, latestDate, c), commits
        )

        commitRelatedAuthors = set(
            list(map(lambda c: authorIdExtractor(c.author), commitRelatedCommits))
        )

        # get current related authors collection and update it
        authorRelatedAuthors = allRelatedAuthors.setdefault(author, set())
        authorRelatedAuthors.update(commitRelatedAuthors)

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
            communityCommitCount = sum(authorCommits[author] for author in community)
            row = [authorCount, communityCommitCount]
            modularity.append(row)
    except ZeroDivisionError:
        # not handled
        pass

    # finding high centrality authors
    numberHighCentralityAuthors = len(
        [author for author, centrality in centrality.items() if centrality > 0.5]
    )

    percentageHighCentralityAuthors = numberHighCentralityAuthors / len(
        allRelatedAuthors
    )

    print("Outputting CSVs")

    # output non-tabular results
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Density", density])
        w.writerow(["Community Count", len(modularity)])

    # output community information
    with open(os.path.join(outputDir, "community.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Community Index", "Author Count", "Commit Count"])
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
    with open(os.path.join(outputDir, "centrality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["Author", "Closeness", "Betweenness", "Centrality"])
        w.writeheader()

        for key in combined:
            w.writerow(combined[key])

    # output high centrality authors
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["NumberHighCentralityAuthors", numberHighCentralityAuthors])
        w.writerow(["PercentageHighCentralityAuthors", percentageHighCentralityAuthors])

    # output statistics
    outputStatistics(
        [value for key, value in closeness.items()], "Closeness", outputDir,
    )

    outputStatistics(
        [value for key, value in betweenness.items()], "Betweenness", outputDir,
    )

    outputStatistics(
        [value for key, value in centrality.items()], "Centrality", outputDir,
    )

    outputStatistics(
        [community[0] for community in modularity], "CommunityAuthorCount", outputDir,
    )

    outputStatistics(
        [community[1] for community in modularity], "CommunityCommitCount", outputDir,
    )

    # output graph to PNG
    print("Outputting graph to PNG")
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
    graphFigure.savefig(os.path.join(outputDir, "graph.png"))


# helper functions
def findRelatedCommits(author, earliestDate, latestDate, commit):
    isDifferentAuthor = author != authorIdExtractor(commit.author)
    if not isDifferentAuthor:
        return False

    commitDate = datetime.fromtimestamp(commit.committed_date)
    isInRange = commitDate >= earliestDate and commitDate <= latestDate
    return isInRange

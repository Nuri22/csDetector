import git
import yaml
import os
import requests
import sys
import re

from configuration import Configuration, parseAliasArgs
from repoLoader import getRepo
from progress.bar import Bar
from utils import authorIdExtractor
from strsimpy.metric_lcs import MetricLCS


def main():
    try:
        # parse args
        config = parseAliasArgs(sys.argv)

        # get repository reference
        repo = getRepo(config)

        # build path
        aliasPath = os.path.join(config.repositoryPath, "aliases.yml")

        # delete existing alias file if present
        if os.path.exists(aliasPath):
            os.remove(aliasPath)

        # extract aliases
        extractAliases(config, repo, aliasPath)

    finally:
        # close repo to avoid resource leaks
        if "repo" in locals():
            del repo


def extractAliases(config: Configuration, repo: git.Repo, aliasPath: str):
    commits = list(repo.iter_commits())

    # get all distinct author emails
    emails = set(
        authorIdExtractor(commit.author) for commit in Bar("Processing").iter(commits)
    )

    # get a commit per email
    shasByEmail = {}
    for email in Bar("Processing").iter(emails):

        commit = next(
            commit
            for commit in repo.iter_commits()
            if authorIdExtractor(commit.author) == email
        )

        shasByEmail[email] = commit.hexsha

    # query github for author logins by their commits
    loginsByEmail = dict()
    emailsWithoutLogins = []

    for email in Bar("Processing").iter(shasByEmail):
        sha = shasByEmail[email]
        url = "https://api.github.com/repos/{}/{}/commits/{}".format(
            config.repositoryOwner, config.repositoryName, sha
        )
        request = requests.get(url, headers={"Authorization": "token " + config.pat})
        commit = request.json()

        if not "author" in commit.keys():
            continue

        if not commit["author"] is None and not commit["author"]["login"] is None:
            loginsByEmail[email] = commit["author"]["login"]
        else:
            emailsWithoutLogins.append(email)

    # build initial alias collection from logins
    aliases = {}
    usedAsValues = {}

    for email in loginsByEmail:
        login = loginsByEmail[email]
        aliasEmails = aliases.setdefault(login, [])
        aliasEmails.append(email)
        usedAsValues[email] = login

    if len(emailsWithoutLogins) > 0:
        for authorA in Bar("Processing").iter(emailsWithoutLogins):
            quickMatched = False

            # go through used values
            for key in usedAsValues:
                if authorA == key:
                    quickMatched = True
                    continue

                if areSimilar(authorA, key, config.maxDistance):
                    alias = usedAsValues[key]
                    aliases[alias].append(authorA)
                    usedAsValues[authorA] = alias
                    quickMatched = True
                    break

            if quickMatched:
                continue

            # go through already extracted keys
            for key in aliases:
                if authorA == key:
                    quickMatched = True
                    continue

                if areSimilar(authorA, key, config.maxDistance):
                    aliases[key].append(authorA)
                    usedAsValues[authorA] = key
                    quickMatched = True
                    break

            if quickMatched:
                continue

            # go through all authors
            for authorB in emailsWithoutLogins:
                if authorA == authorB:
                    continue

                if areSimilar(authorA, authorB, config.maxDistance):
                    aliasedAuthor = aliases.setdefault(authorA, [])
                    aliasedAuthor.append(authorB)
                    usedAsValues[authorB] = authorA
                    break

    print("Writing aliases to '{0}'".format(aliasPath))
    if not os.path.exists(os.path.dirname(aliasPath)):
        os.makedirs(os.path.dirname(aliasPath))

    with open(aliasPath, "a", newline="") as f:
        yaml.dump(aliases, f)


def areSimilar(valueA: str, valueB: str, maxDistance: float):
    lcs = MetricLCS()
    expr = r"(.+)@"

    localPartAMatches = re.findall(expr, valueA)
    localPartBMatches = re.findall(expr, valueB)

    if len(localPartAMatches) == 0:
        localPartAMatches = [valueA]

    if len(localPartBMatches) == 0:
        localPartBMatches = [valueB]

    distance = lcs.distance(localPartAMatches[0], localPartBMatches[0])

    return distance <= maxDistance


main()

import git
import csv
import os

from dateutil.relativedelta import relativedelta
from typing import List
from progress.bar import Bar
from datetime import datetime
from utils import authorIdExtractor
from statsAnalysis import outputStatistics
from sentistrength import PySentiStr
from git.objects.commit import Commit
from configuration import Configuration
import pytz

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def commitAnalysis(
    senti: PySentiStr,
    commits: List[git.Commit],
    delta: relativedelta,
    config: Configuration,
):

    # sort commits
    commits.sort(key=lambda o: o.committed_datetime)

    # split commits into batches
    batches = []
    batch = []
    startDate = None
    if config.startDate is not None:
        startDate = datetime.strptime(config.startDate, "%Y-%m-%d")
        startDate = startDate.replace(tzinfo=pytz.UTC)
    batchStartDate = None
    batchEndDate = None
    batchDates = []

    for commit in Bar("Batching commits").iter(commits):
        if startDate is not None and startDate > commit.committed_datetime:
            continue
        # prepare first batch
        if batchStartDate == None:
            batchStartDate = commit.committed_datetime
            batchEndDate = batchStartDate + delta

            batchDates.append(batchStartDate)

        # prepare next batch
        elif commit.committed_datetime > batchEndDate:
            batches.append(batch)
            batch = []
            batchStartDate = commit.committed_datetime
            batchEndDate = batchStartDate + delta

            batchDates.append(batchStartDate)

        # populate current batch
        batch.append(commit)

    # complete batch list and perform clean up
    batches.append(batch)
    del batch, commits

    # run analysis per batch
    authorInfoDict = {}
    daysActive = list()
    for idx, batch in enumerate(batches):

        # get batch authors
        batchAuthorInfoDict, batchDaysActive = commitBatchAnalysis(
            idx, senti, batch, config
        )

        # combine with main lists
        authorInfoDict.update(batchAuthorInfoDict)
        daysActive.append(batchDaysActive)

    return batchDates, authorInfoDict, daysActive


def commitBatchAnalysis(
    idx: int, senti: PySentiStr, commits: List[git.Commit], config: Configuration
):

    authorInfoDict = {}
    timezoneInfoDict = {}
    experienceDays = 150

    # traverse all commits
    print("Analyzing commits")
    startDate = None
    if config.startDate is not None:
        startDate = datetime.strptime(config.startDate, "%Y-%m-%d")
        startDate = startDate.replace(tzinfo=pytz.UTC)
    # sort commits
    commits.sort(key=lambda o: o.committed_datetime, reverse=True)

    commitMessages = []
    commit: Commit
    lastDate = None
    firstDate = None
    realCommitCount = 0
    if len(commits) > 0:
        for commit in Bar("Processing").iter(commits):
            if startDate is not None and startDate > commit.committed_datetime:
                continue
            if lastDate is None:
                lastDate = commit.committed_date
            firstDate = commit.committed_date
            realCommitCount = realCommitCount + 1
            # extract info
            author = authorIdExtractor(commit.author)
            timezone = commit.author_tz_offset
            time = commit.authored_datetime

            # get timezone
            timezoneInfo = timezoneInfoDict.setdefault(
                timezone, dict(commitCount=0, authors=set())
            )

            # save info
            timezoneInfo["authors"].add(author)

            if commit.message and commit.message.strip():
                commitMessages.append(commit.message)

            # increase commit count
            timezoneInfo["commitCount"] += 1

            # get author
            authorInfo = authorInfoDict.setdefault(
                author,
                dict(
                    commitCount=0,
                    sponsoredCommitCount=0,
                    earliestCommitDate=time,
                    latestCommitDate=time,
                    sponsored=False,
                    activeDays=0,
                    experienced=False,
                ),
            )

            # increase commit count
            authorInfo["commitCount"] += 1

            # validate earliest commit
            # by default GitPython orders commits from latest to earliest
            if time < authorInfo["earliestCommitDate"]:
                authorInfo["earliestCommitDate"] = time

            # check if commit was between 9 and 5
            if not commit.author_tz_offset == 0 and time.hour >= 9 and time.hour <= 17:
                authorInfo["sponsoredCommitCount"] += 1

    print("Analyzing commit message sentiment")
    sentimentScores = []
    commitMessageSentimentsPositive = []
    commitMessageSentimentsNegative = []

    if len(commitMessages) > 0:
        sentimentScores = senti.getSentiment(commitMessages)
        commitMessageSentimentsPositive = list(
            result for result in filter(lambda value: value >= 1, sentimentScores)
        )
        commitMessageSentimentsNegative = list(
            result for result in filter(lambda value: value <= -1, sentimentScores)
        )

    print("Analyzing authors")
    sponsoredAuthorCount = 0
    for login, author in authorInfoDict.items():

        # check if sponsored
        commitCount = int(author["commitCount"])
        sponsoredCommitCount = int(author["sponsoredCommitCount"])
        diff = sponsoredCommitCount / commitCount
        if diff >= 0.95:
            author["sponsored"] = True
            sponsoredAuthorCount += 1

        # calculate active days
        earliestDate = author["earliestCommitDate"]
        latestDate = author["latestCommitDate"]
        activeDays = (latestDate - earliestDate).days + 1
        author["activeDays"] = activeDays

        # check if experienced
        if activeDays >= experienceDays:
            author["experienced"] = True

    # calculate percentage sponsored authors
    percentageSponsoredAuthors = 0
    if len([*authorInfoDict]) > 0:
        percentageSponsoredAuthors = sponsoredAuthorCount / len([*authorInfoDict])

    # calculate active project days
    firstCommitDate = None
    lastCommitDate = None
    if firstDate is not None:
        firstCommitDate = datetime.fromtimestamp(firstDate)
    if lastDate is not None:
        lastCommitDate = datetime.fromtimestamp(lastDate)
    daysActive = 0
    if lastCommitDate is not None:
        daysActive = (lastCommitDate - firstCommitDate).days

    print("Outputting CSVs")

    # output author days on project
    with open(
        os.path.join(config.metricsPath, f"authorDaysOnProject_{idx}.csv"),
        "a",
        newline="",
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Author", "# of Days"])
        for login, author in authorInfoDict.items():
            w.writerow([login, author["activeDays"]])

    # output commits per author
    with open(
        os.path.join(config.metricsPath, f"commitsPerAuthor_{idx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Author", "Commit Count"])
        for login, author in authorInfoDict.items():
            w.writerow([login, author["commitCount"]])

    # output timezones
    with open(
        os.path.join(config.metricsPath, f"timezones_{idx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Timezone Offset", "Author Count", "Commit Count"])
        for key, timezone in timezoneInfoDict.items():
            w.writerow([key, len(timezone["authors"]), timezone["commitCount"]])

    # output results
    with open(
        os.path.join(config.resultsPath, f"results_{idx}.csv"), "a", newline=""
    ) as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["CommitCount", realCommitCount])
        w.writerow(["DaysActive", daysActive])
        if firstCommitDate is not None:
            w.writerow(["FirstCommitDate", "{:%Y-%m-%d}".format(firstCommitDate)])
            w.writerow(["LastCommitDate", "{:%Y-%m-%d}".format(lastCommitDate)])
        else:
            w.writerow(["FirstCommitDate", "00-00-00"])
            w.writerow(["LastCommitDate", "00-00-00"])
        w.writerow(["AuthorCount", len([*authorInfoDict])])
        w.writerow(["SponsoredAuthorCount", sponsoredAuthorCount])
        w.writerow(["PercentageSponsoredAuthors", percentageSponsoredAuthors])
        w.writerow(["TimezoneCount", len([*timezoneInfoDict])])

    outputStatistics(
        idx,
        [author["activeDays"] for login, author in authorInfoDict.items()],
        "AuthorActiveDays",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        [author["commitCount"] for login, author in authorInfoDict.items()],
        "AuthorCommitCount",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        [len(timezone["authors"]) for key, timezone in timezoneInfoDict.items()],
        "TimezoneAuthorCount",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        [timezone["commitCount"] for key, timezone in timezoneInfoDict.items()],
        "TimezoneCommitCount",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        sentimentScores,
        "CommitMessageSentiment",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        commitMessageSentimentsPositive,
        "CommitMessageSentimentsPositive",
        config.resultsPath,
    )

    outputStatistics(
        idx,
        commitMessageSentimentsNegative,
        "CommitMessageSentimentsNegative",
        config.resultsPath,
    )

    return authorInfoDict, daysActive

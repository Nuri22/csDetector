import git
import csv
import os

from typing import List
from progress.bar import Bar
from datetime import datetime
from utils import authorIdExtractor
from statsAnalysis import outputStatistics
from sentistrength import PySentiStr
from git.objects.commit import Commit

def commitAnalysis(commits: List[git.Commit], outputDir: str):

    authorInfoDict = {}
    timezoneInfoDict = {}
    experienceDays = 150

    # traverse all commits
    print("Analyzing commits")
    for commit in Bar("Processing").iter(commits):

        # extract info
        author = authorIdExtractor(commit.author)
        timezone = commit.author_tz_offset
        time = commit.authored_datetime

        # get timezone
        timezoneInfo = timezoneInfoDict.setdefault(
            timezone, dict(commitCount=0, authors=set())
        )

        # save author
        timezoneInfo["authors"].add(author)

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
    sentimentResult = senti.getSentiment(commitMessages)
    commitMessageSentimentsPositive = sum(
        1 for _ in filter(lambda value: value >= 1, sentimentResult)
    )
    commitMessageSentimentsNegative = sum(
        1 for _ in filter(lambda value: value <= -1, sentimentResult)
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
    percentageSponsoredAuthors = sponsoredAuthorCount / len([*authorInfoDict])

    # calculate active project days
    firstCommitDate = datetime.fromtimestamp(commits[len(commits) - 1].committed_date)
    lastCommitDate = datetime.fromtimestamp(commits[0].committed_date)
    daysActive = (lastCommitDate - firstCommitDate).days

    print("Outputting CSVs")

    # output author days on project
    with open(os.path.join(outputDir, "authorDaysOnProject.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Author", "# of Days"])
        for login, author in authorInfoDict.items():
            w.writerow([login, author["activeDays"]])

    # output commits per author
    with open(os.path.join(outputDir, "commitsPerAuthor.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Author", "Commit Count"])
        for login, author in authorInfoDict.items():
            w.writerow([login, author["commitCount"]])

    # output timezones
    with open(os.path.join(outputDir, "timezones.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["Timezone Offset", "Author Count", "Commit Count"])
        for key, timezone in timezoneInfoDict.items():
            w.writerow([key, len(timezone["authors"]), timezone["commitCount"]])

    # output project info
    with open(os.path.join(outputDir, "project.csv"), "a", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["DaysActive", daysActive])
        w.writerow(["FirstCommitDate", "{:%Y-%m-%d}".format(firstCommitDate)])
        w.writerow(["LastCommitDate", "{:%Y-%m-%d}".format(lastCommitDate)])
        w.writerow(["AuthorCount", len([*authorInfoDict])])
        w.writerow(["SponsoredAuthorCount", sponsoredAuthorCount])
        w.writerow(["PercentageSponsoredAuthors", percentageSponsoredAuthors])
        w.writerow(["TimezoneCount", len([*timezoneInfoDict])])
		w.writerow(["CommitMessageSentimentsPositive", commitMessageSentimentsPositive])
        w.writerow(["CommitMessageSentimentsNegative", commitMessageSentimentsNegative])

    outputStatistics(
        [author["activeDays"] for login, author in authorInfoDict.items()],
        "ProjectActiveDays",
        outputDir,
    )

    outputStatistics(
        [author["commitCount"] for login, author in authorInfoDict.items()],
        "ProjectCommitCount",
        outputDir,
    )

    outputStatistics(
        [len(timezone["authors"]) for key, timezone in timezoneInfoDict.items()],
        "TimezoneAuthorCount",
        outputDir,
    )

    outputStatistics(
        [timezone["commitCount"] for key, timezone in timezoneInfoDict.items()],
        "TimezoneCommitCount",
        outputDir,
    )

    outputStatistics(
        sentimentResult, "CommitMessageSentiment", outputDir,
    )
    return authorInfoDict

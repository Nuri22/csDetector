import os
import argparse
from typing import Sequence


class Configuration:
    def __init__(
        self,
        repositoryUrl: str,
        batchMonths: int,
        outputPath: str,
        sentiStrengthPath: str,
        maxDistance: int,
        pat: str,
        googleKey: str,
    ):
        self.repositoryUrl = repositoryUrl
        self.batchMonths = batchMonths
        self.outputPath = outputPath
        self.sentiStrengthPath = sentiStrengthPath
        self.maxDistance = maxDistance
        self.pat = pat
        self.googleKey = googleKey

        # parse repo name into owner and project name
        split = self.repositoryUrl.split("/")
        self.repositoryOwner = split[3]
        self.repositoryName = split[4]

        # build repo path
        self.repositoryPath = os.path.join(self.outputPath, split[3], split[4])

        # build results path
        self.resultsPath = os.path.join(self.repositoryPath, "results")

        # build metrics path
        self.metricsPath = os.path.join(self.resultsPath, "metrics")


def parseAliasArgs(args: Sequence[str]):

    parser = argparse.ArgumentParser(
        description="Extract commit author aliases from GitHub repositories.",
        epilog="Check README file for more information on running this tool.",
    )

    parser.add_argument(
        "-p",
        "--pat",
        help="GitHub PAT (personal access token) used for querying the GitHub API",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--repositoryUrl",
        help="GitHub repository URL that you want to analyse",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--maxDistance",
        help="""string distance metric
        https://github.com/luozhouyang/python-string-similarity#metric-longest-common-subsequence
        """,
        type=float,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--outputPath",
        help="local directory path for analysis output",
        required=True,
    )

    args = parser.parse_args()
    config = Configuration(
        args.repositoryUrl, 0, args.outputPath, "", args.maxDistance, args.pat, ""
    )

    return config


def parseDevNetworkArgs(args: Sequence[str]):

    parser = argparse.ArgumentParser(
        description="Perform network and statistical analysis on GitHub repositories.",
        epilog="Check README file for more information on running this tool.",
    )

    parser.add_argument(
        "-p",
        "--pat",
        help="GitHub PAT (personal access token) used for querying the GitHub API",
        required=True,
    )

    parser.add_argument(
        "-g",
        "--googleKey",
        help="Google Cloud API Key used for authentication with the Perspective API",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--repositoryUrl",
        help="GitHub repository URL that you want to analyse",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--batchMonths",
        help="Number of months to analyze per batch. Default=9999",
        type=float,
        default=9999,
    )

    parser.add_argument(
        "-s",
        "--sentiStrengthPath",
        help="local directory path to the SentiStregth tool",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--outputPath",
        help="Local directory path for analysis output",
        required=True,
    )

    args = parser.parse_args()
    config = Configuration(
        args.repositoryUrl,
        args.batchMonths,
        args.outputPath,
        args.sentiStrengthPath,
        0,
        args.pat,
        args.googleKey,
    )

    return config

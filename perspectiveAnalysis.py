from datetime import datetime
import json
import time
import requests
import math

from typing import List
from configuration import Configuration


def getToxicityPercentage(config: Configuration, comments: List):

    # estimate completion
    qpsLimit = 1
    buffer = 5
    queryLimit = (qpsLimit * 60) - buffer

    toxicityMinutes = math.ceil(len(comments) / queryLimit)
    print(
        f"    Toxicity per comment, expecting around {toxicityMinutes} minute(s) completion time",
        end="",
    )

    # declare toxicity results store
    toxicResults = 0

    # wait until the next minute
    sleepUntilNextMinute()

    # run analysis
    for idx, comment in enumerate(comments):

        # build request
        url = (
            "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
            + "?key="
            + config.googleKey
        )
        data_dict = {
            "comment": {"text": comment},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
        }

        # send request
        response = requests.post(url=url, data=json.dumps(data_dict))

        # parse response
        dict = json.loads(response.content)

        try:
            toxicity = float(
                dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            )
        except:
            print()
            e = dict["error"]
            raise Exception(f'Error {e["code"]} {e["status"]}: {e["message"]}')

        # add to results store if toxic
        if toxicity >= 0.5:
            toxicResults += 1

        print(".", end="")

        # we are only allowed 1 QPS, wait for a minute
        if (idx + 1) % queryLimit == 0:
            print()
            print("        QPS limit reached, napping", end="")
            sleepUntilNextMinute()
            print(", processing", end="")

    print()

    # calculate percentage of toxic comments
    percentage = toxicResults / len(comments)

    return percentage


def sleepUntilNextMinute():
    t = datetime.utcnow()
    sleeptime = 60 - (t.second + t.microsecond / 1000000.0)
    time.sleep(sleeptime)
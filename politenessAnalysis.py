import os
import csv
import convokit

from configuration import Configuration


def politenessAnalysis(
    config: Configuration,
    outputPrefix: str,
    commentBatches: list,
):

    for batchIdx, batch in enumerate(commentBatches):

        # analyze batch
        positiveMarkerCount = getResults(batch)

        # output results
        with open(
            os.path.join(config.resultsPath, f"results_{batchIdx}.csv"),
            "a",
            newline="",
        ) as f:
            w = csv.writer(f, delimiter=",")
            w.writerow([f"RPC{outputPrefix}", positiveMarkerCount])


def getResults(comments: list):

    # define default speaker
    speaker = convokit.Speaker(id="default", name="default")

    # build utterance list
    utterances = list(
        [
            convokit.Utterance(id=str(idx), speaker=speaker, text=comment)
            for idx, comment in enumerate(comments)
        ]
    )

    # build corpus
    corpus = convokit.Corpus(utterances=utterances)

    # parse
    parser = convokit.TextParser(verbosity=1000)
    corpus = parser.transform(corpus)

    # extract politeness features
    politeness = convokit.PolitenessStrategies()
    corpus = politeness.transform(corpus, markers=True)
    features = corpus.get_utterances_dataframe()

    # get positive politeness marker count
    positiveMarkerCount = sum(
        [
            feature["feature_politeness_==HASPOSITIVE=="]
            for feature in features["meta.politeness_strategies"]
        ]
    )

    return positiveMarkerCount

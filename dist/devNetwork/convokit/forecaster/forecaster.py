from convokit.model import Corpus, Conversation, Utterance
from typing import Callable, Optional
from convokit import Transformer
from .cumulativeBoW import CumulativeBoW
from .forecasterModel import ForecasterModel
import pandas as pd

class Forecaster(Transformer):
    """
    Implements basic Forecaster behavior.

    :param forecaster_model: ForecasterModel to use, e.g. cumulativeBoW or CRAFT
    :param forecast_mode: 'future' or 'past'. 'future' (the default behavior) annotates each utterance with a forecast score using all context up to and including that utterance (i.e., a prediction of the future state of the conversation after this utterance). 'past' annotates each utterance with a forecast score using all context prior to that utterance (i.e., what the model believed this utterance would look like prior to actually seeing it)
    :param convo_structure: conversations in expected corpus are 'branched' or 'linear', default: "branched"
    :param text_func: optional function for extracting the text of the utterance, default: uses utterance's text attribute
    :param label_func: callable function for getting the utterance's forecast label (True or False); only used in training
    :param use_last_only: if forecast_mode is 'past' and use_last_only is True, for each dialog, use only the context-reply pair where the reply is the last utterance in the dialog
    :param skip_broken_convos: if True and convo_structure is 'branched', exclude all conversations that have broken reply-to structures, default: True
    :param forecast_attribute_name: metadata feature name to use in annotation for forecast result, default: "forecast"
    :param forecast_prob_attribute_name: metadata feature name to use in annotation for forecast result probability, default: "forecast_prob"
    """
    def __init__(self, forecaster_model: ForecasterModel = None,
                 forecast_mode: str = "future",
                 convo_structure: str = "branched",
                 text_func=lambda utt: utt.text,
                 label_func: Callable[[Utterance], bool] = lambda utt: True,
                 use_last_only: bool = False,
                 skip_broken_convos: bool = True,
                 forecast_attribute_name: str = "forecast",
                 forecast_prob_attribute_name: str = "forecast_prob"
                 ):

        assert convo_structure in ["branched", "linear"]
        self.convo_structure = convo_structure

        if forecaster_model is None:
            print("No model passed to Forecaster. Initializing default forecaster model: Cumulative Bag-of-words...")
            self.forecaster_model = CumulativeBoW(forecast_attribute_name=forecast_attribute_name,
                                                  forecast_prob_attribute_name=forecast_prob_attribute_name)
        else:
            self.forecaster_model = forecaster_model
        self.forecast_mode = forecast_mode
        self.label_func = label_func
        self.text_func = text_func
        self.use_last_only = use_last_only
        self.skip_broken_convos = skip_broken_convos
        self.forecast_attribute_name = forecast_attribute_name
        self.forecast_prob_attribute_name = forecast_prob_attribute_name

    def _get_context_reply_label_dict(self, corpus: Corpus, convo_selector, utt_excluder, include_label=True):
        """
        Returns a dict mapping reply id to (context, reply, label).

        If self.forecast_mode == 'future': return a dict mapping the leaf utt id to the path from root utt to leaf utt
        """
        dialogs = []
        if self.convo_structure == "branched":
            for convo in corpus.iter_conversations(convo_selector):
                try:
                    for path in convo.get_root_to_leaf_paths():
                        path = [utt for utt in path if not utt_excluder(utt)]
                        if len(path) == 1: continue
                        dialogs.append(path)
                except ValueError as e:
                    if not self.skip_broken_convos:
                        raise e

        elif self.convo_structure == "linear":
            for convo in corpus.iter_conversations(convo_selector):
                utts = convo.get_chronological_utterance_list(selector=lambda x: not utt_excluder(x))
                if len(utts) == 1: continue
                dialogs.append(utts)

        id_to_context_reply_label = dict()

        # this flag determines whether the dictionary entry for each utterance ID should include that
        # utterance in the context (True corresponds to "future" behavior). This needs to be always
        # False when include_label = True, since include_label assumes that the label comes from the
        # utterance after the last utterance in the context. This override logic won't affect
        # forecast_mode however, since that argument only applies to transform() while include_label
        # is only True when called from fit()
        include_current = (self.forecast_mode == 'future') and (not include_label)

        for dialog in dialogs:
            if self.use_last_only:
                reply = self.text_func(dialog[-1])
                context = [self.text_func(utt) for utt in (dialog if include_current else dialog[:-1])]
                label = self.label_func(dialog[-1]) if include_label else None
                id_to_context_reply_label[dialog[-1].id] = (context, reply, label)
            else:
                for idx in range(0 if include_current else 1, len(dialog)):
                    reply = self.text_func(dialog[idx])
                    label = self.label_func(dialog[idx]) if include_label else None
                    reply_id = dialog[idx].id
                    context = [self.text_func(utt) for utt in (dialog[:(idx+1)] if include_current else dialog[:idx])]
                    id_to_context_reply_label[reply_id] = (context, reply, label) if include_label else (context, reply, None)

        return id_to_context_reply_label

    def fit(self, corpus: Corpus, y=None,
            selector: Callable[[Conversation], bool] = lambda convo: True,
            ignore_utterances: Callable[[Utterance], bool] = lambda utt: False):
        """
        Train the ForecasterModel on the given corpus.

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Conversation and returns a bool: True if the Conversation is to be included in the fitting step. By default, includes all Conversations.
        :param ignore_utterances: a (lambda) function that takes an Utterance and returns a bool: True if the Utterance should be excluded from the Conversation in the fitting step. By default, all Utterances are included.
        :return: fitted Forecaster Transformer
        """
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, selector, ignore_utterances, include_label=True)
        self.forecaster_model.train(id_to_context_reply_label)

    def transform(self, corpus: Corpus,
                  selector: Callable[[Conversation], bool] = lambda convo: True,
                  ignore_utterances: Callable[[Utterance], bool] = lambda utt: False) -> Corpus:
        """
        Annotate the corpus utterances with forecast and forecast score information

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Conversation and returns a bool: True if the Conversation is to be included in the transformation step. By default, includes all Conversations.
        :param ignore_utterances: a (lambda) function that takes an Utterance and returns a bool: True if the Utterance should be excluded from the Conversation in the transformation step. By default, all Utterances are included.
        :return: annotated Corpus
        """
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, selector, ignore_utterances, include_label=False)
        forecast_df = self.forecaster_model.forecast(id_to_context_reply_label)

        for utt in corpus.iter_utterances():
            if utt.id in forecast_df.index:
                utt.add_meta(self.forecast_attribute_name, forecast_df.loc[utt.id][self.forecast_attribute_name])
                utt.add_meta(self.forecast_prob_attribute_name, forecast_df.loc[utt.id][self.forecast_prob_attribute_name])
            else:
                utt.add_meta(self.forecast_attribute_name, None)
                utt.add_meta(self.forecast_prob_attribute_name, None)

        return corpus

    def fit_transform(self, corpus: Corpus, y=None, selector: Callable[[Conversation], bool] = lambda convo: True,
                      ignore_utterances: Callable[[Utterance], bool] = lambda utt: False) -> Corpus:
        self.fit(corpus, selector=selector, ignore_utterances=ignore_utterances)
        return self.transform(corpus, selector=selector, ignore_utterances=ignore_utterances)

    def summarize(self, corpus: Corpus,
                  selector: Callable[[Conversation], bool] = lambda convo: True,
                  ignore_utterances: Callable[[Utterance], bool] = lambda utt: False,
                  exclude_na=True):
        """
        Returns a DataFrame of utterances and their forecasts (and forecast probabilities)

        :param corpus: target Corpus
        :param exclude_na: whether to drop NaN results
        :param selector: a (lambda) function that takes a Conversation and returns a bool: True if the Conversation is to be included in the summary step. By default, includes all Conversations.
        :param ignore_utterances: a (lambda) function that takes an Utterance and returns a bool: True if the Utterance should be excluded from the Conversation in the summary step. By default, all Utterances are included.
        :return: a pandas DataFrame
        """
        utt_forecast_prob = []
        for convo in corpus.iter_conversations(selector):
            for utt in convo.iter_utterances(lambda x: not ignore_utterances(x)):
                utt_forecast_prob.append((utt.id, utt.meta[self.forecast_attribute_name], utt.meta[self.forecast_prob_attribute_name]))
        forecast_df = pd.DataFrame(utt_forecast_prob, columns=["utt_id", self.forecast_attribute_name, self.forecast_prob_attribute_name]) \
            .set_index('utt_id').sort_values(self.forecast_prob_attribute_name, ascending=False)
        if exclude_na:
            forecast_df = forecast_df.dropna()
        return forecast_df

    def get_model(self):
        """
        Get the forecaster model object
        """
        return self.forecaster_model

    def set_model(self, forecaster_model):
        """
        Set the forecaster model
        :return:
        """
        self.forecaster_model = forecaster_model

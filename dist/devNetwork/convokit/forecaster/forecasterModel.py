from abc import ABC, abstractmethod
from convokit.util import deprecation

class ForecasterModel(ABC):

    def __init__(self, forecast_attribute_name: str = "prediction",
                 forecast_feat_name=None,
                 forecast_prob_attribute_name: str = "score",
                 forecast_prob_feat_name=None):
        """

        :param forecast_attribute_name: name for DataFrame column containing predictions, default: "prediction"
        :param forecast_prob_attribute_name: name for column containing prediction scores, default: "score"
        """
        self.forecast_attribute_name = forecast_attribute_name if forecast_feat_name is None else forecast_feat_name
        self.forecast_prob_attribute_name = forecast_prob_attribute_name if forecast_prob_feat_name is None else\
            forecast_prob_feat_name

        for deprecated_set in [(forecast_feat_name, 'forecast_feat_name', 'forecast_attribute_name'),
                               (forecast_prob_feat_name, 'forecast_prob_feat_name', 'forecast_prob_attribute_name')]:
            if deprecated_set[0] is not None:
                deprecation(f"Forecaster's {deprecated_set[1]} parameter", f'{deprecated_set[2]}')

    @abstractmethod
    def train(self, id_to_context_reply_label):
        """
        Train the Forecaster Model with the context-reply-label tuples
        """
        pass

    @abstractmethod
    def forecast(self, id_to_context_reply_label):
        """
        Use the Forecaster Model to compute forecasts and scores
        for given context-reply pairs and return a results dataframe
        """
        pass



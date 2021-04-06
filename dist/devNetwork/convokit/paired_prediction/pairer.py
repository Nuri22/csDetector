from typing import Callable
from .util import *
from collections import defaultdict
from random import shuffle, choice

from convokit.util import deprecation
from convokit import Transformer, CorpusComponent, Corpus


class Pairer(Transformer):
    """
    The Pairer Transformer annotates the Corpus with the pairing information that is needed to run some paired prediction analysis (e.g. see the PairedPrediction and Paired BoW modules.)

    Pairer sets this pairing up. For this example:

    - The obj_type is 'utterance' (since we compare utterances)
    - The pairing_func is supposed to extract the identifier that would identify the object as part of the pair. In this case, that would be the Utterance's conversation id since we want utterances from the same conversation.
    - We need to distinguish between utterances where Rachel speaks to Monica vs. Chandler.
        the pos_label_func and neg_label_func is how we can specify this (e.g. lambda utt: utt.meta['target']), where positive instances might be arbitrarily refer to targetting Monica, and negative for targetting Chandler.
    - pair_mode denotes how many pairs to use per context.
        For example, a Conversation will likely have Rachel address Monica and Chandler each multiple times.
        This means that there are multiple positive and negative instances that can be used to form pairs.
        We could randomly pick one pair of instances ('random'), or the first pair of instances ('first'),
        or the maximum pairs of instances ('maximize').

    Pairer saves this pairing information into the object metadata.

    - pair_id is the 'id' that uniquely identifies a pair of positive and negative instances,
        and is the output from the pairing_func.
    - label (or pair_obj_label) denotes whether the object is the positive or negative instance of the pair
    - pair_orientation denotes whether to use the pair itself as a positive or negative data point in a predictive
        classifier. 'pos' means the difference between the objects in the pair should be computed as
        [+ve obj features] - [-ve obj features], and 'neg' means it should be computed as
        [-ve obj features] - [+ve obj features].
    """

    def __init__(self, obj_type: str,
                 pairing_func: Callable[[CorpusComponent], str],
                 pos_label_func: Callable[[CorpusComponent], bool],
                 neg_label_func: Callable[[CorpusComponent], bool],
                 pair_mode: str = "random",
                 pair_id_attribute_name: str = "pair_id",
                 pair_id_feat_name=None,
                 label_attribute_name: str = "pair_obj_label",
                 label_feat_name=None,
                 pair_orientation_attribute_name: str = "pair_orientation",
                 pair_orientation_feat_name=None
                 ):

        """
        :param pairing_func: the Corpus object characteristic to pair on, e.g. to pair on the first 10 characters of a
            well-structured id, use lambda obj: obj.id[:10]
        :param pos_label_func: The function to check if the object is a positive instance
        :param neg_label_func: The function to check if the object is a negative instance
        :param pair_mode: 'random': pick a single positive and negative object pair randomly (default), 'maximize': pick the maximum number of positive and negative object pairs possible randomly, or 'first': pick the first positive and negative object pair found.
        :param pair_id_attribute_name: metadata attribute name to use in annotating object with pair id, default: "pair_id". The value is determined by the output of pairing_func. If pair_mode is 'maximize', the value is the output of pairing_func + "_[i]", where i is the ith pair extracted from a given context.
        :param label_attribute_name: metadata attribute name to use in annotating object with whether it is positive or negative, default: "pair_obj_label"
        :param pair_orientation_attribute_name: metadata attribute name to use in annotating object with pair orientation, default: "pair_orientation"

        """
        assert obj_type in ["speaker", "utterance", "conversation"]
        self.obj_type = obj_type
        self.pairing_func = pairing_func
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.pair_mode = pair_mode
        self.pair_id_attribute_name = pair_id_attribute_name if pair_id_feat_name is None else pair_id_feat_name
        self.label_attribute_name = label_attribute_name if label_feat_name is None else label_feat_name
        self.pair_orientation_attribute_name = pair_orientation_attribute_name if \
            pair_orientation_feat_name is None else pair_orientation_feat_name

        for deprecated_set in [(pair_id_feat_name, 'pair_id_feat_name', 'pair_id_attribute_name'),
                               (label_feat_name, 'label_feat_name', 'label_attribute_name'),
                               (pair_orientation_feat_name, 'pair_orientation_feat_name',
                                'pair_orientation_attribute_name')]:
            if deprecated_set[0] is not None:
                deprecation(f"Pairer's {deprecated_set[1]} parameter", f'{deprecated_set[2]}')

    def _get_pos_neg_objects(self, corpus: Corpus, selector):
        """
        Get positively-labelled and negatively-labelled lists of objects

        :param corpus: target Corpus
        :return: list of positive objects, list of negative objects
        """
        pos_objects = []
        neg_objects = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            if self.pos_label_func(obj):
                pos_objects.append(obj)
            elif self.neg_label_func(obj):
                neg_objects.append(obj)
        return pos_objects, neg_objects

    def _pair_objs(self, pos_objects, neg_objects):
        """
        Generate a dictionary mapping the Corpus object characteristic value (i.e. pairing_func's output) to one positively and negatively labelled object.

        :param pos_objects: list of positively labelled objects
        :param neg_objects: list of negatively labelled objects
        :return: dictionary indexed by the paired feature instance value,
                 with the value being a tuple (pos_obj, neg_obj)
        """
        pair_feat_to_pos_objs = defaultdict(list)
        pair_feat_to_neg_objs = defaultdict(list)

        for obj in pos_objects:
            pair_feat_to_pos_objs[self.pairing_func(obj)].append(obj)

        for obj in neg_objects:
            pair_feat_to_neg_objs[self.pairing_func(obj)].append(obj)

        valid_pairs = set(pair_feat_to_neg_objs).intersection(set(pair_feat_to_pos_objs))

        if self.pair_mode == "first":
            return {pair_id: (pair_feat_to_pos_objs[pair_id][0],
                              pair_feat_to_neg_objs[pair_id][0])
                    for pair_id in valid_pairs}
        elif self.pair_mode == "random":
            return {pair_id: (choice(pair_feat_to_pos_objs[pair_id]),
                              choice(pair_feat_to_neg_objs[pair_id]))
                    for pair_id in valid_pairs}
        elif self.pair_mode == "maximize":
            retval = dict()
            for pair_id in valid_pairs:
                pos_objs = pair_feat_to_pos_objs[pair_id]
                neg_objs = pair_feat_to_neg_objs[pair_id]
                max_pairs = min(len(pos_objs), len(neg_objs))
                shuffle(pos_objs)
                shuffle(neg_objs)
                for idx in range(max_pairs):
                    retval[pair_id + "_" + str(idx)] = (pos_objs[idx], neg_objs[idx])
            return retval
        else:
            raise ValueError("Invalid pair_mode setting: use 'random', 'first', or 'maximize'.")

    @staticmethod
    def _assign_pair_orientations(obj_pairs):
        """
        Assigns the pair orientation (i.e. whether this pair will have a positive or negative label)

        :param obj_pairs: dictionary indexed by the paired feature instance value
        :return: dictionary of paired feature instance values to pair orientation value ('pos' or 'neg')
        """
        pair_ids = list(obj_pairs)
        shuffle(pair_ids)
        pair_orientations = dict()
        flip = True
        for pair_id in pair_ids:
            pair_orientations[pair_id] = "pos" if flip else "neg"
            flip = not flip
        return pair_orientations

    def transform(self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True) -> Corpus:
        """
        Annotate corpus objects with pair information (label, pair_id, pair_orientation), with an optional selector indicating which objects should be considered for pairing.

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns a bool (True = include)
        :return: annotated Corpus
        """
        pos_objs, neg_objs = self._get_pos_neg_objects(corpus, selector)
        obj_pairs = self._pair_objs(pos_objs, neg_objs)
        pair_orientations = self._assign_pair_orientations(obj_pairs)

        for pair_id, (pos_obj, neg_obj) in obj_pairs.items():
            pos_obj.add_meta(self.label_attribute_name, "pos")
            neg_obj.add_meta(self.label_attribute_name, "neg")
            pos_obj.add_meta(self.pair_id_attribute_name, pair_id)
            neg_obj.add_meta(self.pair_id_attribute_name, pair_id)
            pos_obj.add_meta(self.pair_orientation_attribute_name, pair_orientations[pair_id])
            neg_obj.add_meta(self.pair_orientation_attribute_name, pair_orientations[pair_id])

        for obj in corpus.iter_objs(self.obj_type):
            # unlabelled objects include both objects that did not pass the selector
            # and objects that were not selected in the pairing step
            if self.label_attribute_name not in obj.meta:
                obj.add_meta(self.label_attribute_name, None)
                obj.add_meta(self.pair_id_attribute_name, None)
                obj.add_meta(self.pair_orientation_attribute_name, None)

        return corpus
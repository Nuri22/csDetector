import pytest
import random
import numpy.random
from numpy.testing import assert_equal
from thinc.api import fix_random_seed
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.tokens import Doc
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.training import Example

from ..util import make_tempdir


TRAIN_DATA_SINGLE_LABEL = [
    ("I'm so happy.", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("I'm so angry", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]

TRAIN_DATA_MULTI_LABEL = [
    ("I'm angry and confused", {"cats": {"ANGRY": 1.0, "CONFUSED": 1.0, "HAPPY": 0.0}}),
    ("I'm confused but happy", {"cats": {"ANGRY": 0.0, "CONFUSED": 1.0, "HAPPY": 1.0}}),
]


def make_get_examples_single_label(nlp):
    train_examples = []
    for t in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    def get_examples():
        return train_examples

    return get_examples


def make_get_examples_multi_label(nlp):
    train_examples = []
    for t in TRAIN_DATA_MULTI_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    def get_examples():
        return train_examples

    return get_examples


@pytest.mark.skip(reason="Test is flakey when run with others")
def test_simple_train():
    nlp = Language()
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("answer")
    nlp.initialize()
    for i in range(5):
        for text, answer in [
            ("aaaa", 1.0),
            ("bbbb", 0),
            ("aa", 1.0),
            ("bbbbbbbbb", 0.0),
            ("aaaaaa", 1),
        ]:
            nlp.update((text, {"cats": {"answer": answer}}))
    doc = nlp("aaa")
    assert "answer" in doc.cats
    assert doc.cats["answer"] >= 0.5


@pytest.mark.skip(reason="Test is flakey when run with others")
def test_textcat_learns_multilabel():
    random.seed(5)
    numpy.random.seed(5)
    docs = []
    nlp = Language()
    letters = ["a", "b", "c"]
    for w1 in letters:
        for w2 in letters:
            cats = {letter: float(w2 == letter) for letter in letters}
            docs.append((Doc(nlp.vocab, words=["d"] * 3 + [w1, w2] + ["d"] * 3), cats))
    random.shuffle(docs)
    textcat = TextCategorizer(nlp.vocab, width=8)
    for letter in letters:
        textcat.add_label(letter)
    optimizer = textcat.initialize(lambda: [])
    for i in range(30):
        losses = {}
        examples = [Example.from_dict(doc, {"cats": cats}) for doc, cat in docs]
        textcat.update(examples, sgd=optimizer, losses=losses)
        random.shuffle(docs)
    for w1 in letters:
        for w2 in letters:
            doc = Doc(nlp.vocab, words=["d"] * 3 + [w1, w2] + ["d"] * 3)
            truth = {letter: w2 == letter for letter in letters}
            textcat(doc)
            for cat, score in doc.cats.items():
                if not truth[cat]:
                    assert score < 0.5
                else:
                    assert score > 0.5


@pytest.mark.parametrize("name", ["textcat", "textcat_multilabel"])
def test_label_types(name):
    nlp = Language()
    textcat = nlp.add_pipe(name)
    textcat.add_label("answer")
    with pytest.raises(ValueError):
        textcat.add_label(9)


@pytest.mark.parametrize("name", ["textcat", "textcat_multilabel"])
def test_no_label(name):
    nlp = Language()
    nlp.add_pipe(name)
    with pytest.raises(ValueError):
        nlp.initialize()


@pytest.mark.parametrize(
    "name,get_examples",
    [
        ("textcat", make_get_examples_single_label),
        ("textcat_multilabel", make_get_examples_multi_label),
    ],
)
def test_implicit_label(name, get_examples):
    nlp = Language()
    nlp.add_pipe(name)
    nlp.initialize(get_examples=get_examples(nlp))


@pytest.mark.parametrize("name", ["textcat", "textcat_multilabel"])
def test_no_resize(name):
    nlp = Language()
    textcat = nlp.add_pipe(name)
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")
    nlp.initialize()
    assert textcat.model.get_dim("nO") >= 2
    # this throws an error because the textcat can't be resized after initialization
    with pytest.raises(ValueError):
        textcat.add_label("NEUTRAL")


def test_error_with_multi_labels():
    nlp = Language()
    nlp.add_pipe("textcat")
    train_examples = []
    for text, annotations in TRAIN_DATA_MULTI_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    with pytest.raises(ValueError):
        nlp.initialize(get_examples=lambda: train_examples)


@pytest.mark.parametrize(
    "name,get_examples, train_data",
    [
        ("textcat", make_get_examples_single_label, TRAIN_DATA_SINGLE_LABEL),
        ("textcat_multilabel", make_get_examples_multi_label, TRAIN_DATA_MULTI_LABEL),
    ],
)
def test_initialize_examples(name, get_examples, train_data):
    nlp = Language()
    textcat = nlp.add_pipe(name)
    for text, annotations in train_data:
        for label, value in annotations.get("cats").items():
            textcat.add_label(label)
    # you shouldn't really call this more than once, but for testing it should be fine
    nlp.initialize()
    nlp.initialize(get_examples=get_examples(nlp))
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: None)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=get_examples())


def test_overfitting_IO():
    # Simple test to try and quickly overfit the single-label textcat component - ensuring the ML models work correctly
    fix_random_seed(0)
    nlp = English()
    textcat = nlp.add_pipe("textcat")

    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert textcat.model.get_dim("nO") == 2

    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["textcat"] < 0.01

    # test the trained model
    test_text = "I am happy."
    doc = nlp(test_text)
    cats = doc.cats
    assert cats["POSITIVE"] > 0.9
    assert cats["POSITIVE"] + cats["NEGATIVE"] == pytest.approx(1.0, 0.001)

    # Also test the results are still the same after IO
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        cats2 = doc2.cats
        assert cats2["POSITIVE"] > 0.9
        assert cats2["POSITIVE"] + cats2["NEGATIVE"] == pytest.approx(1.0, 0.001)

    # Test scoring
    scores = nlp.evaluate(train_examples)
    assert scores["cats_micro_f"] == 1.0
    assert scores["cats_macro_f"] == 1.0
    assert scores["cats_macro_auc"] == 1.0
    assert scores["cats_score"] == 1.0
    assert "cats_score_desc" in scores

    # Make sure that running pipe twice, or comparing to call, always amounts to the same predictions
    texts = ["Just a sentence.", "I like green eggs.", "I am happy.", "I eat ham."]
    batch_cats_1 = [doc.cats for doc in nlp.pipe(texts)]
    batch_cats_2 = [doc.cats for doc in nlp.pipe(texts)]
    no_batch_cats = [doc.cats for doc in [nlp(text) for text in texts]]
    assert_equal(batch_cats_1, batch_cats_2)
    assert_equal(batch_cats_1, no_batch_cats)


def test_overfitting_IO_multi():
    # Simple test to try and quickly overfit the multi-label textcat component - ensuring the ML models work correctly
    fix_random_seed(0)
    nlp = English()
    textcat = nlp.add_pipe("textcat_multilabel")

    train_examples = []
    for text, annotations in TRAIN_DATA_MULTI_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    assert textcat.model.get_dim("nO") == 3

    for i in range(100):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["textcat_multilabel"] < 0.01

    # test the trained model
    test_text = "I am confused but happy."
    doc = nlp(test_text)
    cats = doc.cats
    assert cats["HAPPY"] > 0.9
    assert cats["CONFUSED"] > 0.9

    # Also test the results are still the same after IO
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        cats2 = doc2.cats
        assert cats2["HAPPY"] > 0.9
        assert cats2["CONFUSED"] > 0.9

    # Test scoring
    scores = nlp.evaluate(train_examples)
    assert scores["cats_micro_f"] == 1.0
    assert scores["cats_macro_f"] == 1.0
    assert "cats_score_desc" in scores

    # Make sure that running pipe twice, or comparing to call, always amounts to the same predictions
    texts = ["Just a sentence.", "I like green eggs.", "I am happy.", "I eat ham."]
    batch_deps_1 = [doc.cats for doc in nlp.pipe(texts)]
    batch_deps_2 = [doc.cats for doc in nlp.pipe(texts)]
    no_batch_deps = [doc.cats for doc in [nlp(text) for text in texts]]
    assert_equal(batch_deps_1, batch_deps_2)
    assert_equal(batch_deps_1, no_batch_deps)


# fmt: off
@pytest.mark.parametrize(
    "name,train_data,textcat_config",
    [
        ("textcat_multilabel", TRAIN_DATA_MULTI_LABEL, {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": False, "ngram_size": 1, "no_output_layer": False}),
        ("textcat", TRAIN_DATA_SINGLE_LABEL, {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": True, "ngram_size": 4, "no_output_layer": False}),
        ("textcat_multilabel", TRAIN_DATA_MULTI_LABEL, {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": False, "ngram_size": 3, "no_output_layer": True}),
        ("textcat", TRAIN_DATA_SINGLE_LABEL, {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": True, "ngram_size": 2, "no_output_layer": True}),
        ("textcat_multilabel", TRAIN_DATA_MULTI_LABEL, {"@architectures": "spacy.TextCatEnsemble.v2", "tok2vec": DEFAULT_TOK2VEC_MODEL, "linear_model": {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": False, "ngram_size": 1, "no_output_layer": False}}),
        ("textcat", TRAIN_DATA_SINGLE_LABEL, {"@architectures": "spacy.TextCatEnsemble.v2", "tok2vec": DEFAULT_TOK2VEC_MODEL, "linear_model": {"@architectures": "spacy.TextCatBOW.v1", "exclusive_classes": True, "ngram_size": 5, "no_output_layer": False}}),
        ("textcat", TRAIN_DATA_SINGLE_LABEL, {"@architectures": "spacy.TextCatCNN.v1", "tok2vec": DEFAULT_TOK2VEC_MODEL, "exclusive_classes": True}),
        ("textcat_multilabel", TRAIN_DATA_MULTI_LABEL, {"@architectures": "spacy.TextCatCNN.v1", "tok2vec": DEFAULT_TOK2VEC_MODEL, "exclusive_classes": False}),
    ],
)
# fmt: on
def test_textcat_configs(name, train_data, textcat_config):
    pipe_config = {"model": textcat_config}
    nlp = English()
    textcat = nlp.add_pipe(name, config=pipe_config)
    train_examples = []
    for text, annotations in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for label, value in annotations.get("cats").items():
            textcat.add_label(label)
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)


def test_positive_class():
    nlp = English()
    textcat = nlp.add_pipe("textcat")
    get_examples = make_get_examples_single_label(nlp)
    textcat.initialize(get_examples, labels=["POS", "NEG"], positive_label="POS")
    assert textcat.labels == ("POS", "NEG")
    assert textcat.cfg["positive_label"] == "POS"

    textcat_multilabel = nlp.add_pipe("textcat_multilabel")
    get_examples = make_get_examples_multi_label(nlp)
    with pytest.raises(TypeError):
        textcat_multilabel.initialize(
            get_examples, labels=["POS", "NEG"], positive_label="POS"
        )
    textcat_multilabel.initialize(get_examples, labels=["FICTION", "DRAMA"])
    assert textcat_multilabel.labels == ("FICTION", "DRAMA")
    assert "positive_label" not in textcat_multilabel.cfg


def test_positive_class_not_present():
    nlp = English()
    textcat = nlp.add_pipe("textcat")
    get_examples = make_get_examples_single_label(nlp)
    with pytest.raises(ValueError):
        textcat.initialize(get_examples, labels=["SOME", "THING"], positive_label="POS")


def test_positive_class_not_binary():
    nlp = English()
    textcat = nlp.add_pipe("textcat")
    get_examples = make_get_examples_multi_label(nlp)
    with pytest.raises(ValueError):
        textcat.initialize(
            get_examples, labels=["SOME", "THING", "POS"], positive_label="POS"
        )


def test_textcat_evaluation():
    train_examples = []
    nlp = English()
    ref1 = nlp("one")
    ref1.cats = {"winter": 1.0, "summer": 1.0, "spring": 1.0, "autumn": 1.0}
    pred1 = nlp("one")
    pred1.cats = {"winter": 1.0, "summer": 0.0, "spring": 1.0, "autumn": 1.0}
    train_examples.append(Example(pred1, ref1))

    ref2 = nlp("two")
    ref2.cats = {"winter": 0.0, "summer": 0.0, "spring": 1.0, "autumn": 1.0}
    pred2 = nlp("two")
    pred2.cats = {"winter": 1.0, "summer": 0.0, "spring": 0.0, "autumn": 1.0}
    train_examples.append(Example(pred2, ref2))

    scores = Scorer().score_cats(
        train_examples, "cats", labels=["winter", "summer", "spring", "autumn"]
    )
    assert scores["cats_f_per_type"]["winter"]["p"] == 1 / 2
    assert scores["cats_f_per_type"]["winter"]["r"] == 1 / 1
    assert scores["cats_f_per_type"]["summer"]["p"] == 0
    assert scores["cats_f_per_type"]["summer"]["r"] == 0 / 1
    assert scores["cats_f_per_type"]["spring"]["p"] == 1 / 1
    assert scores["cats_f_per_type"]["spring"]["r"] == 1 / 2
    assert scores["cats_f_per_type"]["autumn"]["p"] == 2 / 2
    assert scores["cats_f_per_type"]["autumn"]["r"] == 2 / 2

    assert scores["cats_micro_p"] == 4 / 5
    assert scores["cats_micro_r"] == 4 / 6


def test_textcat_threshold():
    # Ensure the scorer can be called with a different threshold
    nlp = English()
    nlp.add_pipe("textcat")

    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    nlp.initialize(get_examples=lambda: train_examples)

    # score the model (it's not actually trained but that doesn't matter)
    scores = nlp.evaluate(train_examples)
    assert 0 <= scores["cats_score"] <= 1

    scores = nlp.evaluate(train_examples, scorer_cfg={"threshold": 1.0})
    assert scores["cats_f_per_type"]["POSITIVE"]["r"] == 0

    scores = nlp.evaluate(train_examples, scorer_cfg={"threshold": 0})
    macro_f = scores["cats_score"]
    assert scores["cats_f_per_type"]["POSITIVE"]["r"] == 1.0

    scores = nlp.evaluate(train_examples, scorer_cfg={"threshold": 0, "positive_label": "POSITIVE"})
    pos_f = scores["cats_score"]
    assert scores["cats_f_per_type"]["POSITIVE"]["r"] == 1.0
    assert pos_f > macro_f


def test_textcat_multi_threshold():
    # Ensure the scorer can be called with a different threshold
    nlp = English()
    nlp.add_pipe("textcat_multilabel")

    train_examples = []
    for text, annotations in TRAIN_DATA_SINGLE_LABEL:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    nlp.initialize(get_examples=lambda: train_examples)

    # score the model (it's not actually trained but that doesn't matter)
    scores = nlp.evaluate(train_examples)
    assert 0 <= scores["cats_score"] <= 1

    scores = nlp.evaluate(train_examples, scorer_cfg={"threshold": 1.0})
    assert scores["cats_f_per_type"]["POSITIVE"]["r"] == 0

    scores = nlp.evaluate(train_examples, scorer_cfg={"threshold": 0})
    assert scores["cats_f_per_type"]["POSITIVE"]["r"] == 1.0

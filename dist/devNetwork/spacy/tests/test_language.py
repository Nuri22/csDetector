import itertools
import logging
from unittest import mock
import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from spacy.training import Example
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.util import registry, ignore_error, raise_error
import spacy

from .util import add_vecs_to_vocab, assert_docs_equal


@pytest.fixture
def nlp():
    nlp = Language(Vocab())
    textcat = nlp.add_pipe("textcat")
    for label in ("POSITIVE", "NEGATIVE"):
        textcat.add_label(label)
    nlp.initialize()
    return nlp


def test_language_update(nlp):
    text = "hello world"
    annots = {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}
    wrongkeyannots = {"LABEL": True}
    doc = Doc(nlp.vocab, words=text.split(" "))
    example = Example.from_dict(doc, annots)
    nlp.update([example])

    # Not allowed to call with just one Example
    with pytest.raises(TypeError):
        nlp.update(example)

    # Update with text and dict: not supported anymore since v.3
    with pytest.raises(TypeError):
        nlp.update((text, annots))
    # Update with doc object and dict
    with pytest.raises(TypeError):
        nlp.update((doc, annots))

    # Create examples badly
    with pytest.raises(ValueError):
        example = Example.from_dict(doc, None)
    with pytest.raises(KeyError):
        example = Example.from_dict(doc, wrongkeyannots)


def test_language_evaluate(nlp):
    text = "hello world"
    annots = {"doc_annotation": {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}}
    doc = Doc(nlp.vocab, words=text.split(" "))
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    assert scores["speed"] > 0

    # test with generator
    scores = nlp.evaluate(eg for eg in [example])
    assert scores["speed"] > 0

    # Not allowed to call with just one Example
    with pytest.raises(TypeError):
        nlp.evaluate(example)

    # Evaluate with text and dict: not supported anymore since v.3
    with pytest.raises(TypeError):
        nlp.evaluate([(text, annots)])
    # Evaluate with doc object and dict
    with pytest.raises(TypeError):
        nlp.evaluate([(doc, annots)])
    with pytest.raises(TypeError):
        nlp.evaluate([text, annots])


def test_evaluate_no_pipe(nlp):
    """Test that docs are processed correctly within Language.pipe if the
    component doesn't expose a .pipe method."""

    @Language.component("test_evaluate_no_pipe")
    def pipe(doc):
        return doc

    text = "hello world"
    annots = {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}
    nlp = Language(Vocab())
    doc = nlp(text)
    nlp.add_pipe("test_evaluate_no_pipe")
    nlp.evaluate([Example.from_dict(doc, annots)])


@Language.component("test_language_vector_modification_pipe")
def vector_modification_pipe(doc):
    doc.vector += 1
    return doc


@Language.component("test_language_userdata_pipe")
def userdata_pipe(doc):
    doc.user_data["foo"] = "bar"
    return doc


@Language.component("test_language_ner_pipe")
def ner_pipe(doc):
    span = Span(doc, 0, 1, label="FIRST")
    doc.ents += (span,)
    return doc


@pytest.fixture
def sample_vectors():
    return [
        ("spacy", [-0.1, -0.2, -0.3]),
        ("world", [-0.2, -0.3, -0.4]),
        ("pipe", [0.7, 0.8, 0.9]),
    ]


@pytest.fixture
def nlp2(nlp, sample_vectors):
    add_vecs_to_vocab(nlp.vocab, sample_vectors)
    nlp.add_pipe("test_language_vector_modification_pipe")
    nlp.add_pipe("test_language_ner_pipe")
    nlp.add_pipe("test_language_userdata_pipe")
    return nlp


@pytest.fixture
def texts():
    data = [
        "Hello world.",
        "This is spacy.",
        "You can use multiprocessing with pipe method.",
        "Please try!",
    ]
    return data


@pytest.mark.parametrize("n_process", [1, 2])
def test_language_pipe(nlp2, n_process, texts):
    texts = texts * 10
    expecteds = [nlp2(text) for text in texts]
    docs = nlp2.pipe(texts, n_process=n_process, batch_size=2)

    for doc, expected_doc in zip(docs, expecteds):
        assert_docs_equal(doc, expected_doc)


@pytest.mark.parametrize("n_process", [1, 2])
def test_language_pipe_stream(nlp2, n_process, texts):
    # check if nlp.pipe can handle infinite length iterator properly.
    stream_texts = itertools.cycle(texts)
    texts0, texts1 = itertools.tee(stream_texts)
    expecteds = (nlp2(text) for text in texts0)
    docs = nlp2.pipe(texts1, n_process=n_process, batch_size=2)

    n_fetch = 20
    for doc, expected_doc in itertools.islice(zip(docs, expecteds), n_fetch):
        assert_docs_equal(doc, expected_doc)


def test_language_pipe_error_handler():
    """Test that the error handling of nlp.pipe works well"""
    nlp = English()
    nlp.add_pipe("merge_subtokens")
    nlp.initialize()
    texts = ["Curious to see what will happen to this text.", "And this one."]
    # the pipeline fails because there's no parser
    with pytest.raises(ValueError):
        nlp(texts[0])
    with pytest.raises(ValueError):
        list(nlp.pipe(texts))
    nlp.set_error_handler(raise_error)
    with pytest.raises(ValueError):
        list(nlp.pipe(texts))
    # set explicitely to ignoring
    nlp.set_error_handler(ignore_error)
    docs = list(nlp.pipe(texts))
    assert len(docs) == 0
    nlp(texts[0])


def test_language_pipe_error_handler_custom(en_vocab):
    """Test the error handling of a custom component that has no pipe method"""

    @Language.component("my_evil_component")
    def evil_component(doc):
        if "2" in doc.text:
            raise ValueError("no dice")
        return doc

    def warn_error(proc_name, proc, docs, e):
        from spacy.util import logger

        logger.warning(f"Trouble with component {proc_name}.")

    nlp = English()
    nlp.add_pipe("my_evil_component")
    nlp.initialize()
    texts = ["TEXT 111", "TEXT 222", "TEXT 333", "TEXT 342", "TEXT 666"]
    with pytest.raises(ValueError):
        # the evil custom component throws an error
        list(nlp.pipe(texts))

    nlp.set_error_handler(warn_error)
    logger = logging.getLogger("spacy")
    with mock.patch.object(logger, "warning") as mock_warning:
        # the errors by the evil custom component raise a warning for each bad batch
        docs = list(nlp.pipe(texts))
        mock_warning.assert_called()
        assert mock_warning.call_count == 2
        assert len(docs) + mock_warning.call_count == len(texts)
        assert [doc.text for doc in docs] == ["TEXT 111", "TEXT 333", "TEXT 666"]


def test_language_pipe_error_handler_pipe(en_vocab):
    """Test the error handling of a component's pipe method"""

    @Language.component("my_sentences")
    def perhaps_set_sentences(doc):
        if not doc.text.startswith("4"):
            doc[-1].is_sent_start = True
        return doc

    texts = [f"{str(i)} is enough. Done" for i in range(100)]
    nlp = English()
    nlp.add_pipe("my_sentences")
    entity_linker = nlp.add_pipe("entity_linker", config={"entity_vector_length": 3})
    entity_linker.kb.add_entity(entity="Q1", freq=12, entity_vector=[1, 2, 3])
    nlp.initialize()
    with pytest.raises(ValueError):
        # the entity linker requires sentence boundaries, will throw an error otherwise
        docs = list(nlp.pipe(texts, batch_size=10))
    nlp.set_error_handler(ignore_error)
    docs = list(nlp.pipe(texts, batch_size=10))
    # we lose/ignore the failing 0-9 and 40-49 batches
    assert len(docs) == 80


def test_language_from_config_before_after_init():
    name = "test_language_from_config_before_after_init"
    ran_before = False
    ran_after = False
    ran_after_pipeline = False
    ran_before_init = False
    ran_after_init = False

    @registry.callbacks(f"{name}_before")
    def make_before_creation():
        def before_creation(lang_cls):
            nonlocal ran_before
            ran_before = True
            assert lang_cls is English
            lang_cls.Defaults.foo = "bar"
            return lang_cls

        return before_creation

    @registry.callbacks(f"{name}_after")
    def make_after_creation():
        def after_creation(nlp):
            nonlocal ran_after
            ran_after = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == []
            assert nlp.Defaults.foo == "bar"
            nlp.meta["foo"] = "bar"
            return nlp

        return after_creation

    @registry.callbacks(f"{name}_after_pipeline")
    def make_after_pipeline_creation():
        def after_pipeline_creation(nlp):
            nonlocal ran_after_pipeline
            ran_after_pipeline = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == ["sentencizer"]
            assert nlp.Defaults.foo == "bar"
            assert nlp.meta["foo"] == "bar"
            nlp.meta["bar"] = "baz"
            return nlp

        return after_pipeline_creation

    @registry.callbacks(f"{name}_before_init")
    def make_before_init():
        def before_init(nlp):
            nonlocal ran_before_init
            ran_before_init = True
            nlp.meta["before_init"] = "before"
            return nlp

        return before_init

    @registry.callbacks(f"{name}_after_init")
    def make_after_init():
        def after_init(nlp):
            nonlocal ran_after_init
            ran_after_init = True
            nlp.meta["after_init"] = "after"
            return nlp

        return after_init

    config = {
        "nlp": {
            "pipeline": ["sentencizer"],
            "before_creation": {"@callbacks": f"{name}_before"},
            "after_creation": {"@callbacks": f"{name}_after"},
            "after_pipeline_creation": {"@callbacks": f"{name}_after_pipeline"},
        },
        "components": {"sentencizer": {"factory": "sentencizer"}},
        "initialize": {
            "before_init": {"@callbacks": f"{name}_before_init"},
            "after_init": {"@callbacks": f"{name}_after_init"},
        },
    }
    nlp = English.from_config(config)
    assert nlp.Defaults.foo == "bar"
    assert nlp.meta["foo"] == "bar"
    assert nlp.meta["bar"] == "baz"
    assert "before_init" not in nlp.meta
    assert "after_init" not in nlp.meta
    assert nlp.pipe_names == ["sentencizer"]
    assert nlp("text")
    nlp.initialize()
    assert nlp.meta["before_init"] == "before"
    assert nlp.meta["after_init"] == "after"
    assert all(
        [ran_before, ran_after, ran_after_pipeline, ran_before_init, ran_after_init]
    )


def test_language_from_config_before_after_init_invalid():
    """Check that an error is raised if function doesn't return nlp."""
    name = "test_language_from_config_before_after_init_invalid"
    registry.callbacks(f"{name}_before1", func=lambda: lambda nlp: None)
    registry.callbacks(f"{name}_before2", func=lambda: lambda nlp: nlp())
    registry.callbacks(f"{name}_after1", func=lambda: lambda nlp: None)
    registry.callbacks(f"{name}_after1", func=lambda: lambda nlp: English)

    for callback_name in [f"{name}_before1", f"{name}_before2"]:
        config = {"nlp": {"before_creation": {"@callbacks": callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)
    for callback_name in [f"{name}_after1", f"{name}_after2"]:
        config = {"nlp": {"after_creation": {"@callbacks": callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)
    for callback_name in [f"{name}_after1", f"{name}_after2"]:
        config = {"nlp": {"after_pipeline_creation": {"@callbacks": callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)


def test_language_custom_tokenizer():
    """Test that a fully custom tokenizer can be plugged in via the registry."""
    name = "test_language_custom_tokenizer"

    class CustomTokenizer:
        """Dummy "tokenizer" that splits on spaces and adds prefix to each word."""

        def __init__(self, nlp, prefix):
            self.vocab = nlp.vocab
            self.prefix = prefix

        def __call__(self, text):
            words = [f"{self.prefix}{word}" for word in text.split(" ")]
            return Doc(self.vocab, words=words)

    @registry.tokenizers(name)
    def custom_create_tokenizer(prefix: str = "_"):
        def create_tokenizer(nlp):
            return CustomTokenizer(nlp, prefix=prefix)

        return create_tokenizer

    config = {"nlp": {"tokenizer": {"@tokenizers": name}}}
    nlp = English.from_config(config)
    doc = nlp("hello world")
    assert [t.text for t in doc] == ["_hello", "_world"]
    doc = list(nlp.pipe(["hello world"]))[0]
    assert [t.text for t in doc] == ["_hello", "_world"]


def test_language_from_config_invalid_lang():
    """Test that calling Language.from_config raises an error and lang defined
    in config needs to match language-specific subclasses."""
    config = {"nlp": {"lang": "en"}}
    with pytest.raises(ValueError):
        Language.from_config(config)
    with pytest.raises(ValueError):
        German.from_config(config)


def test_spacy_blank():
    nlp = spacy.blank("en")
    assert nlp.config["training"]["dropout"] == 0.1
    config = {"training": {"dropout": 0.2}}
    meta = {"name": "my_custom_model"}
    nlp = spacy.blank("en", config=config, meta=meta)
    assert nlp.config["training"]["dropout"] == 0.2
    assert nlp.meta["name"] == "my_custom_model"


@pytest.mark.parametrize("value", [False, None, ["x", "y"], Language, Vocab])
def test_language_init_invalid_vocab(value):
    err_fragment = "invalid value"
    with pytest.raises(ValueError) as e:
        Language(value)
    assert err_fragment in str(e.value)

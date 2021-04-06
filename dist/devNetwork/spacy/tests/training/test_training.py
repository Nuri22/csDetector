import numpy
from spacy.training import offsets_to_biluo_tags, biluo_tags_to_offsets, Alignment
from spacy.training import biluo_tags_to_spans, iob_to_biluo
from spacy.training import Corpus, docs_to_json, Example
from spacy.training.align import get_alignments
from spacy.training.converters import json_to_docs
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces, minibatch
from thinc.api import compounding
import pytest
import srsly

from ..util import make_tempdir


@pytest.fixture
def doc():
    nlp = English()  # make sure we get a new vocab every time
    # fmt: off
    words = ["Sarah", "'s", "sister", "flew", "to", "Silicon", "Valley", "via", "London", "."]
    tags = ["NNP", "POS", "NN", "VBD", "IN", "NNP", "NNP", "IN", "NNP", "."]
    pos = ["PROPN", "PART", "NOUN", "VERB", "ADP", "PROPN", "PROPN", "ADP", "PROPN", "PUNCT"]
    morphs = ["NounType=prop|Number=sing", "Poss=yes", "Number=sing", "Tense=past|VerbForm=fin",
              "", "NounType=prop|Number=sing", "NounType=prop|Number=sing", "",
              "NounType=prop|Number=sing", "PunctType=peri"]
    # head of '.' is intentionally nonprojective for testing
    heads = [2, 0, 3, 3, 3, 6, 4, 3, 7, 5]
    deps = ["poss", "case", "nsubj", "ROOT", "prep", "compound", "pobj", "prep", "pobj", "punct"]
    lemmas = ["Sarah", "'s", "sister", "fly", "to", "Silicon", "Valley", "via", "London", "."]
    ents = ["O"] * len(words)
    ents[0] = "B-PERSON"
    ents[1] = "I-PERSON"
    ents[5] = "B-LOC"
    ents[6] = "I-LOC"
    ents[8] = "B-GPE"
    cats = {"TRAVEL": 1.0, "BAKING": 0.0}
    # fmt: on
    doc = Doc(
        nlp.vocab,
        words=words,
        tags=tags,
        pos=pos,
        morphs=morphs,
        heads=heads,
        deps=deps,
        lemmas=lemmas,
        ents=ents,
    )
    doc.cats = cats
    return doc


@pytest.fixture()
def merged_dict():
    return {
        "ids": [1, 2, 3, 4, 5, 6, 7],
        "words": ["Hi", "there", "everyone", "It", "is", "just", "me"],
        "spaces": [True, True, True, True, True, True, False],
        "tags": ["INTJ", "ADV", "PRON", "PRON", "AUX", "ADV", "PRON"],
        "sent_starts": [1, 0, 0, 1, 0, 0, 0],
    }


@pytest.fixture
def vocab():
    nlp = English()
    return nlp.vocab


def test_gold_biluo_U(en_vocab):
    words = ["I", "flew", "to", "London", "."]
    spaces = [True, True, True, False, True]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [(len("I flew to "), len("I flew to London"), "LOC")]
    tags = offsets_to_biluo_tags(doc, entities)
    assert tags == ["O", "O", "O", "U-LOC", "O"]


def test_gold_biluo_BL(en_vocab):
    words = ["I", "flew", "to", "San", "Francisco", "."]
    spaces = [True, True, True, True, False, True]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [(len("I flew to "), len("I flew to San Francisco"), "LOC")]
    tags = offsets_to_biluo_tags(doc, entities)
    assert tags == ["O", "O", "O", "B-LOC", "L-LOC", "O"]


def test_gold_biluo_BIL(en_vocab):
    words = ["I", "flew", "to", "San", "Francisco", "Valley", "."]
    spaces = [True, True, True, True, True, False, True]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [(len("I flew to "), len("I flew to San Francisco Valley"), "LOC")]
    tags = offsets_to_biluo_tags(doc, entities)
    assert tags == ["O", "O", "O", "B-LOC", "I-LOC", "L-LOC", "O"]


def test_gold_biluo_overlap(en_vocab):
    words = ["I", "flew", "to", "San", "Francisco", "Valley", "."]
    spaces = [True, True, True, True, True, False, True]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [
        (len("I flew to "), len("I flew to San Francisco Valley"), "LOC"),
        (len("I flew to "), len("I flew to San Francisco"), "LOC"),
    ]
    with pytest.raises(ValueError):
        offsets_to_biluo_tags(doc, entities)


def test_gold_biluo_misalign(en_vocab):
    words = ["I", "flew", "to", "San", "Francisco", "Valley."]
    spaces = [True, True, True, True, True, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [(len("I flew to "), len("I flew to San Francisco Valley"), "LOC")]
    with pytest.warns(UserWarning):
        tags = offsets_to_biluo_tags(doc, entities)
    assert tags == ["O", "O", "O", "-", "-", "-"]


def test_example_constructor(en_vocab):
    words = ["I", "like", "stuff"]
    tags = ["NOUN", "VERB", "NOUN"]
    tag_ids = [en_vocab.strings.add(tag) for tag in tags]
    predicted = Doc(en_vocab, words=words)
    reference = Doc(en_vocab, words=words)
    reference = reference.from_array("TAG", numpy.array(tag_ids, dtype="uint64"))
    example = Example(predicted, reference)
    tags = example.get_aligned("TAG", as_string=True)
    assert tags == ["NOUN", "VERB", "NOUN"]


def test_example_from_dict_tags(en_vocab):
    words = ["I", "like", "stuff"]
    tags = ["NOUN", "VERB", "NOUN"]
    predicted = Doc(en_vocab, words=words)
    example = Example.from_dict(predicted, {"TAGS": tags})
    tags = example.get_aligned("TAG", as_string=True)
    assert tags == ["NOUN", "VERB", "NOUN"]


def test_example_from_dict_no_ner(en_vocab):
    words = ["a", "b", "c", "d"]
    spaces = [True, True, False, True]
    predicted = Doc(en_vocab, words=words, spaces=spaces)
    example = Example.from_dict(predicted, {"words": words})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == [None, None, None, None]


def test_example_from_dict_some_ner(en_vocab):
    words = ["a", "b", "c", "d"]
    spaces = [True, True, False, True]
    predicted = Doc(en_vocab, words=words, spaces=spaces)
    example = Example.from_dict(
        predicted, {"words": words, "entities": ["U-LOC", None, None, None]}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["U-LOC", None, None, None]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_json_to_docs_no_ner(en_vocab):
    data = [
        {
            "id": 1,
            "paragraphs": [
                {
                    "sentences": [
                        {
                            "tokens": [
                                {"dep": "nn", "head": 1, "tag": "NNP", "orth": "Ms."},
                                {
                                    "dep": "nsubj",
                                    "head": 1,
                                    "tag": "NNP",
                                    "orth": "Haag",
                                },
                                {
                                    "dep": "ROOT",
                                    "head": 0,
                                    "tag": "VBZ",
                                    "orth": "plays",
                                },
                                {
                                    "dep": "dobj",
                                    "head": -1,
                                    "tag": "NNP",
                                    "orth": "Elianti",
                                },
                                {"dep": "punct", "head": -2, "tag": ".", "orth": "."},
                            ]
                        }
                    ]
                }
            ],
        }
    ]
    docs = list(json_to_docs(data))
    assert len(docs) == 1
    for doc in docs:
        assert not doc.has_annotation("ENT_IOB")
    for token in doc:
        assert token.ent_iob == 0
    eg = Example(
        Doc(
            doc.vocab,
            words=[w.text for w in doc],
            spaces=[bool(w.whitespace_) for w in doc],
        ),
        doc,
    )
    ner_tags = eg.get_aligned_ner()
    assert ner_tags == [None, None, None, None, None]


def test_split_sentences(en_vocab):
    # fmt: off
    words = ["I", "flew", "to", "San Francisco Valley", "had", "loads of fun"]
    gold_words = ["I", "flew", "to", "San", "Francisco", "Valley", "had", "loads", "of", "fun"]
    sent_starts = [True, False, False, False, False, False, True, False, False, False]
    # fmt: on
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {"words": gold_words, "sent_starts": sent_starts})
    assert example.text == "I flew to San Francisco Valley had loads of fun "
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == "I flew to San Francisco Valley "
    assert split_examples[1].text == "had loads of fun "
    # fmt: off
    words = ["I", "flew", "to", "San", "Francisco", "Valley", "had", "loads", "of fun"]
    gold_words = ["I", "flew", "to", "San Francisco", "Valley", "had", "loads of", "fun"]
    sent_starts = [True, False, False, False, False, True, False, False]
    # fmt: on
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {"words": gold_words, "sent_starts": sent_starts})
    assert example.text == "I flew to San Francisco Valley had loads of fun "
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == "I flew to San Francisco Valley "
    assert split_examples[1].text == "had loads of fun "


def test_gold_biluo_one_to_many(en_vocab, en_tokenizer):
    words = ["Mr and ", "Mrs Smith", "flew to", "San Francisco Valley", "."]
    spaces = [True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = "Mr and Mrs Smith flew to "
    entities = [(len(prefix), len(prefix + "San Francisco Valley"), "LOC")]
    gold_words = ["Mr and Mrs Smith", "flew", "to", "San", "Francisco", "Valley", "."]
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "U-LOC", "O"]

    entities = [
        (len("Mr and "), len("Mr and Mrs Smith"), "PERSON"),  # "Mrs Smith" is a PERSON
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    # fmt: off
    gold_words = ["Mr and", "Mrs", "Smith", "flew", "to", "San", "Francisco", "Valley", "."]
    # fmt: on
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "U-PERSON", "O", "U-LOC", "O"]

    entities = [
        (len("Mr and "), len("Mr and Mrs"), "PERSON"),  # "Mrs" is a Person
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    # fmt: off
    gold_words = ["Mr and", "Mrs", "Smith", "flew", "to", "San", "Francisco", "Valley", "."]
    # fmt: on
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", None, "O", "U-LOC", "O"]


def test_gold_biluo_many_to_one(en_vocab, en_tokenizer):
    words = ["Mr and", "Mrs", "Smith", "flew", "to", "San", "Francisco", "Valley", "."]
    spaces = [True, True, True, True, True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = "Mr and Mrs Smith flew to "
    entities = [(len(prefix), len(prefix + "San Francisco Valley"), "LOC")]
    gold_words = ["Mr and Mrs Smith", "flew to", "San Francisco Valley", "."]
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "O", "O", "B-LOC", "I-LOC", "L-LOC", "O"]

    entities = [
        (len("Mr and "), len("Mr and Mrs Smith"), "PERSON"),  # "Mrs Smith" is a PERSON
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    gold_words = ["Mr and", "Mrs Smith", "flew to", "San Francisco Valley", "."]
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    expected = ["O", "B-PERSON", "L-PERSON", "O", "O", "B-LOC", "I-LOC", "L-LOC", "O"]
    assert ner_tags == expected


def test_gold_biluo_misaligned(en_vocab, en_tokenizer):
    words = ["Mr and Mrs", "Smith", "flew", "to", "San Francisco", "Valley", "."]
    spaces = [True, True, True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = "Mr and Mrs Smith flew to "
    entities = [(len(prefix), len(prefix + "San Francisco Valley"), "LOC")]
    gold_words = ["Mr", "and Mrs Smith", "flew to", "San", "Francisco Valley", "."]
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "O", "B-LOC", "L-LOC", "O"]

    entities = [
        (len("Mr and "), len("Mr and Mrs Smith"), "PERSON"),  # "Mrs Smith" is a PERSON
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    gold_words = ["Mr and", "Mrs Smith", "flew to", "San", "Francisco Valley", "."]
    example = Example.from_dict(doc, {"words": gold_words, "entities": entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == [None, None, "O", "O", "B-LOC", "L-LOC", "O"]


def test_gold_biluo_additional_whitespace(en_vocab, en_tokenizer):
    # additional whitespace tokens in GoldParse words
    words, spaces = get_words_and_spaces(
        ["I", "flew", "to", "San Francisco", "Valley", "."],
        "I flew  to San Francisco Valley.",
    )
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = "I flew  to "
    entities = [(len(prefix), len(prefix + "San Francisco Valley"), "LOC")]
    gold_words = ["I", "flew", " ", "to", "San Francisco Valley", "."]
    gold_spaces = [True, True, False, True, False, False]
    example = Example.from_dict(
        doc, {"words": gold_words, "spaces": gold_spaces, "entities": entities}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "O", "B-LOC", "L-LOC", "O"]


def test_gold_biluo_4791(en_vocab, en_tokenizer):
    doc = en_tokenizer("I'll return the ₹54 amount")
    gold_words = ["I", "'ll", "return", "the", "₹", "54", "amount"]
    gold_spaces = [False, True, True, True, False, True, False]
    entities = [(16, 19, "MONEY")]
    example = Example.from_dict(
        doc, {"words": gold_words, "spaces": gold_spaces, "entities": entities}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "O", "U-MONEY", "O"]

    doc = en_tokenizer("I'll return the $54 amount")
    gold_words = ["I", "'ll", "return", "the", "$", "54", "amount"]
    gold_spaces = [False, True, True, True, False, True, False]
    entities = [(16, 19, "MONEY")]
    example = Example.from_dict(
        doc, {"words": gold_words, "spaces": gold_spaces, "entities": entities}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "O", "O", "O", "B-MONEY", "L-MONEY", "O"]


def test_roundtrip_offsets_biluo_conversion(en_tokenizer):
    text = "I flew to Silicon Valley via London."
    biluo_tags = ["O", "O", "O", "B-LOC", "L-LOC", "O", "U-GPE", "O"]
    offsets = [(10, 24, "LOC"), (29, 35, "GPE")]
    doc = en_tokenizer(text)
    biluo_tags_converted = offsets_to_biluo_tags(doc, offsets)
    assert biluo_tags_converted == biluo_tags
    offsets_converted = biluo_tags_to_offsets(doc, biluo_tags)
    offsets_converted = [ent for ent in offsets if ent[2]]
    assert offsets_converted == offsets


def test_biluo_spans(en_tokenizer):
    doc = en_tokenizer("I flew to Silicon Valley via London.")
    biluo_tags = ["O", "O", "O", "B-LOC", "L-LOC", "O", "U-GPE", "O"]
    spans = biluo_tags_to_spans(doc, biluo_tags)
    spans = [span for span in spans if span.label_]
    assert len(spans) == 2
    assert spans[0].text == "Silicon Valley"
    assert spans[0].label_ == "LOC"
    assert spans[1].text == "London"
    assert spans[1].label_ == "GPE"


def test_aligned_spans_y2x(en_vocab, en_tokenizer):
    words = ["Mr and Mrs Smith", "flew", "to", "San Francisco Valley", "."]
    spaces = [True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = "Mr and Mrs Smith flew to "
    entities = [
        (0, len("Mr and Mrs Smith"), "PERSON"),
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    # fmt: off
    tokens_ref = ["Mr", "and", "Mrs", "Smith", "flew", "to", "San", "Francisco", "Valley", "."]
    # fmt: on
    example = Example.from_dict(doc, {"words": tokens_ref, "entities": entities})
    ents_ref = example.reference.ents
    assert [(ent.start, ent.end) for ent in ents_ref] == [(0, 4), (6, 9)]
    ents_y2x = example.get_aligned_spans_y2x(ents_ref)
    assert [(ent.start, ent.end) for ent in ents_y2x] == [(0, 1), (3, 4)]


def test_aligned_spans_x2y(en_vocab, en_tokenizer):
    text = "Mr and Mrs Smith flew to San Francisco Valley"
    nlp = English()
    patterns = [
        {"label": "PERSON", "pattern": "Mr and Mrs Smith"},
        {"label": "LOC", "pattern": "San Francisco Valley"},
    ]
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    doc = nlp(text)
    assert [(ent.start, ent.end) for ent in doc.ents] == [(0, 4), (6, 9)]
    prefix = "Mr and Mrs Smith flew to "
    entities = [
        (0, len("Mr and Mrs Smith"), "PERSON"),
        (len(prefix), len(prefix + "San Francisco Valley"), "LOC"),
    ]
    tokens_ref = ["Mr and Mrs", "Smith", "flew", "to", "San Francisco", "Valley"]
    example = Example.from_dict(doc, {"words": tokens_ref, "entities": entities})
    assert [(ent.start, ent.end) for ent in example.reference.ents] == [(0, 2), (4, 6)]
    # Ensure that 'get_aligned_spans_x2y' has the aligned entities correct
    ents_pred = example.predicted.ents
    assert [(ent.start, ent.end) for ent in ents_pred] == [(0, 4), (6, 9)]
    ents_x2y = example.get_aligned_spans_x2y(ents_pred)
    assert [(ent.start, ent.end) for ent in ents_x2y] == [(0, 2), (4, 6)]


def test_gold_ner_missing_tags(en_tokenizer):
    doc = en_tokenizer("I flew to Silicon Valley via London.")
    biluo_tags = [None, "O", "O", "B-LOC", "L-LOC", "O", "U-GPE", "O"]
    example = Example.from_dict(doc, {"entities": biluo_tags})
    assert example.get_aligned("ENT_IOB") == [0, 2, 2, 3, 1, 2, 3, 2]


def test_projectivize(en_tokenizer):
    doc = en_tokenizer("He pretty quickly walks away")
    heads = [3, 2, 3, 0, 2]
    deps = ["dep"] * len(heads)
    example = Example.from_dict(doc, {"heads": heads, "deps": deps})
    proj_heads, proj_labels = example.get_aligned_parse(projectivize=True)
    nonproj_heads, nonproj_labels = example.get_aligned_parse(projectivize=False)
    assert proj_heads == [3, 2, 3, 0, 3]
    assert nonproj_heads == [3, 2, 3, 0, 2]


def test_iob_to_biluo():
    good_iob = ["O", "O", "B-LOC", "I-LOC", "O", "B-PERSON"]
    good_biluo = ["O", "O", "B-LOC", "L-LOC", "O", "U-PERSON"]
    bad_iob = ["O", "O", '"', "B-LOC", "I-LOC"]
    converted_biluo = iob_to_biluo(good_iob)
    assert good_biluo == converted_biluo
    with pytest.raises(ValueError):
        iob_to_biluo(bad_iob)


def test_roundtrip_docs_to_docbin(doc):
    text = doc.text
    idx = [t.idx for t in doc]
    tags = [t.tag_ for t in doc]
    pos = [t.pos_ for t in doc]
    morphs = [str(t.morph) for t in doc]
    lemmas = [t.lemma_ for t in doc]
    deps = [t.dep_ for t in doc]
    heads = [t.head.i for t in doc]
    cats = doc.cats
    ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    # roundtrip to DocBin
    with make_tempdir() as tmpdir:
        # use a separate vocab to test that all labels are added
        reloaded_nlp = English()
        json_file = tmpdir / "roundtrip.json"
        srsly.write_json(json_file, [docs_to_json(doc)])
        output_file = tmpdir / "roundtrip.spacy"
        DocBin(docs=[doc]).to_disk(output_file)
        reader = Corpus(output_file)
        reloaded_examples = list(reader(reloaded_nlp))
    assert len(doc) == sum(len(eg) for eg in reloaded_examples)
    reloaded_example = reloaded_examples[0]
    assert text == reloaded_example.reference.text
    assert idx == [t.idx for t in reloaded_example.reference]
    assert tags == [t.tag_ for t in reloaded_example.reference]
    assert pos == [t.pos_ for t in reloaded_example.reference]
    assert morphs == [str(t.morph) for t in reloaded_example.reference]
    assert lemmas == [t.lemma_ for t in reloaded_example.reference]
    assert deps == [t.dep_ for t in reloaded_example.reference]
    assert heads == [t.head.i for t in reloaded_example.reference]
    assert ents == [
        (e.start_char, e.end_char, e.label_) for e in reloaded_example.reference.ents
    ]
    assert "TRAVEL" in reloaded_example.reference.cats
    assert "BAKING" in reloaded_example.reference.cats
    assert cats["TRAVEL"] == reloaded_example.reference.cats["TRAVEL"]
    assert cats["BAKING"] == reloaded_example.reference.cats["BAKING"]


@pytest.mark.parametrize(
    "tokens_a,tokens_b,expected",
    [
        (["a", "b", "c"], ["ab", "c"], ([[0], [0], [1]], [[0, 1], [2]])),
        (
            ["a", "b", '"', "c"],
            ['ab"', "c"],
            ([[0], [0], [0], [1]], [[0, 1, 2], [3]]),
        ),
        (["a", "bc"], ["ab", "c"], ([[0], [0, 1]], [[0, 1], [1]])),
        (
            ["ab", "c", "d"],
            ["a", "b", "cd"],
            ([[0, 1], [2], [2]], [[0], [0], [1, 2]]),
        ),
        (
            ["a", "b", "cd"],
            ["a", "b", "c", "d"],
            ([[0], [1], [2, 3]], [[0], [1], [2], [2]]),
        ),
        ([" ", "a"], ["a"], ([[], [0]], [[1]])),
        (
            ["a", "''", "'", ","],
            ["a'", "''", ","],
            ([[0], [0, 1], [1], [2]], [[0, 1], [1, 2], [3]]),
        ),
    ],
)
def test_align(tokens_a, tokens_b, expected):  # noqa
    a2b, b2a = get_alignments(tokens_a, tokens_b)
    assert (a2b, b2a) == expected  # noqa
    # check symmetry
    a2b, b2a = get_alignments(tokens_b, tokens_a)  # noqa
    assert (b2a, a2b) == expected  # noqa


def test_goldparse_startswith_space(en_tokenizer):
    text = " a"
    doc = en_tokenizer(text)
    gold_words = ["a"]
    entities = ["U-DATE"]
    deps = ["ROOT"]
    heads = [0]
    example = Example.from_dict(
        doc, {"words": gold_words, "entities": entities, "deps": deps, "heads": heads}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["O", "U-DATE"]
    assert example.get_aligned("DEP", as_string=True) == [None, "ROOT"]


def test_goldparse_endswith_space(en_tokenizer):
    text = "a\n"
    doc = en_tokenizer(text)
    gold_words = ["a"]
    entities = ["U-DATE"]
    deps = ["ROOT"]
    heads = [0]
    example = Example.from_dict(
        doc, {"words": gold_words, "entities": entities, "deps": deps, "heads": heads}
    )
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ["U-DATE", "O"]
    assert example.get_aligned("DEP", as_string=True) == ["ROOT", None]


def test_gold_constructor():
    """Test that the Example constructor works fine"""
    nlp = English()
    doc = nlp("This is a sentence")
    example = Example.from_dict(doc, {"cats": {"cat1": 1.0, "cat2": 0.0}})
    assert example.get_aligned("ORTH", as_string=True) == [
        "This",
        "is",
        "a",
        "sentence",
    ]
    assert example.reference.cats["cat1"]
    assert not example.reference.cats["cat2"]


def test_tuple_format_implicit():
    """Test tuple format"""

    train_data = [
        ("Uber blew through $1 million a week", {"entities": [(0, 4, "ORG")]}),
        (
            "Spotify steps up Asia expansion",
            {"entities": [(0, 7, "ORG"), (17, 21, "LOC")]},
        ),
        ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]}),
    ]

    _train_tuples(train_data)


def test_tuple_format_implicit_invalid():
    """Test that an error is thrown for an implicit invalid field"""
    train_data = [
        ("Uber blew through $1 million a week", {"frumble": [(0, 4, "ORG")]}),
        (
            "Spotify steps up Asia expansion",
            {"entities": [(0, 7, "ORG"), (17, 21, "LOC")]},
        ),
        ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]}),
    ]
    with pytest.raises(KeyError):
        _train_tuples(train_data)


def _train_tuples(train_data):
    nlp = English()
    ner = nlp.add_pipe("ner")
    ner.add_label("ORG")
    ner.add_label("LOC")
    train_examples = []
    for t in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)


def test_split_sents(merged_dict):
    nlp = English()
    example = Example.from_dict(
        Doc(nlp.vocab, words=merged_dict["words"], spaces=merged_dict["spaces"]),
        merged_dict,
    )
    assert example.text == "Hi there everyone It is just me"
    split_examples = example.split_sents()
    assert len(split_examples) == 2
    assert split_examples[0].text == "Hi there everyone "
    assert split_examples[1].text == "It is just me"
    token_annotation_1 = split_examples[0].to_dict()["token_annotation"]
    assert token_annotation_1["ORTH"] == ["Hi", "there", "everyone"]
    assert token_annotation_1["TAG"] == ["INTJ", "ADV", "PRON"]
    assert token_annotation_1["SENT_START"] == [1, 0, 0]
    token_annotation_2 = split_examples[1].to_dict()["token_annotation"]
    assert token_annotation_2["ORTH"] == ["It", "is", "just", "me"]
    assert token_annotation_2["TAG"] == ["PRON", "AUX", "ADV", "PRON"]
    assert token_annotation_2["SENT_START"] == [1, 0, 0, 0]


def test_alignment():
    other_tokens = ["i", "listened", "to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts", "."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 1, 1]
    assert list(align.y2x.dataXd) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_alignment_case_insensitive():
    other_tokens = ["I", "listened", "to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "Obama", "'s", "PODCASTS", "."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 1, 1]
    assert list(align.y2x.dataXd) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_alignment_complex():
    other_tokens = ["i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [0, 0, 0, 1, 2, 3, 4, 5]


def test_alignment_complex_example(en_vocab):
    other_tokens = ["i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts."]
    predicted = Doc(
        en_vocab, words=other_tokens, spaces=[True, False, False, True, False, False]
    )
    reference = Doc(
        en_vocab, words=spacy_tokens, spaces=[True, True, True, False, True, False]
    )
    assert predicted.text == "i listened to obama's podcasts."
    assert reference.text == "i listened to obama's podcasts."
    example = Example(predicted, reference)
    align = example.alignment
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [0, 0, 0, 1, 2, 3, 4, 5]


def test_alignment_different_texts():
    other_tokens = ["she", "listened", "to", "obama", "'s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts", "."]
    with pytest.raises(ValueError):
        Alignment.from_strings(other_tokens, spacy_tokens)


def test_alignment_spaces(en_vocab):
    # single leading whitespace
    other_tokens = [" ", "i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [0, 3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [1, 1, 1, 2, 3, 4, 5, 6]

    # multiple leading whitespace tokens
    other_tokens = [" ", " ", "i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [0, 0, 3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [2, 2, 2, 3, 4, 5, 6, 7]

    # both with leading whitespace, not identical
    other_tokens = [" ", " ", "i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = [" ", "i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [1, 0, 3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 5, 5, 6, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [0, 2, 2, 2, 3, 4, 5, 6, 7]

    # same leading whitespace, different tokenization
    other_tokens = [" ", " ", "i listened to", "obama", "'", "s", "podcasts", "."]
    spacy_tokens = ["  ", "i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [1, 1, 3, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 0, 1, 2, 3, 4, 5, 5, 6, 6]
    assert list(align.y2x.lengths) == [2, 1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [0, 1, 2, 2, 2, 3, 4, 5, 6, 7]

    # only one with trailing whitespace
    other_tokens = ["i listened to", "obama", "'", "s", "podcasts", ".", " "]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts."]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1, 0]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2]
    assert list(align.y2x.dataXd) == [0, 0, 0, 1, 2, 3, 4, 5]

    # different trailing whitespace
    other_tokens = ["i listened to", "obama", "'", "s", "podcasts", ".", " ", " "]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts.", " "]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1, 1, 0]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2, 1]
    assert list(align.y2x.dataXd) == [0, 0, 0, 1, 2, 3, 4, 5, 6]

    # same trailing whitespace, different tokenization
    other_tokens = ["i listened to", "obama", "'", "s", "podcasts", ".", " ", " "]
    spacy_tokens = ["i", "listened", "to", "obama", "'s", "podcasts.", "  "]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.lengths) == [3, 1, 1, 1, 1, 1, 1, 1]
    assert list(align.x2y.dataXd) == [0, 1, 2, 3, 4, 4, 5, 5, 6, 6]
    assert list(align.y2x.lengths) == [1, 1, 1, 1, 2, 2, 2]
    assert list(align.y2x.dataXd) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]

    # differing whitespace is allowed
    other_tokens = ["a", " \n ", "b", "c"]
    spacy_tokens = ["a", "b", " ", "c"]
    align = Alignment.from_strings(other_tokens, spacy_tokens)
    assert list(align.x2y.dataXd) == [0, 1, 3]
    assert list(align.y2x.dataXd) == [0, 2, 3]

    # other differences in whitespace are allowed
    other_tokens = [" ", "a"]
    spacy_tokens = ["  ", "a", " "]
    align = Alignment.from_strings(other_tokens, spacy_tokens)

    other_tokens = ["a", " "]
    spacy_tokens = ["a", "  "]
    align = Alignment.from_strings(other_tokens, spacy_tokens)


def test_retokenized_docs(doc):
    a = doc.to_array(["TAG"])
    doc1 = Doc(doc.vocab, words=[t.text for t in doc]).from_array(["TAG"], a)
    doc2 = Doc(doc.vocab, words=[t.text for t in doc]).from_array(["TAG"], a)
    example = Example(doc1, doc2)
    # fmt: off
    expected1 = ["Sarah", "'s", "sister", "flew", "to", "Silicon", "Valley", "via", "London", "."]
    expected2 = [None, "sister", "flew", "to", None, "via", "London", "."]
    # fmt: on
    assert example.get_aligned("ORTH", as_string=True) == expected1
    with doc1.retokenize() as retokenizer:
        retokenizer.merge(doc1[0:2])
        retokenizer.merge(doc1[5:7])
    assert example.get_aligned("ORTH", as_string=True) == expected2

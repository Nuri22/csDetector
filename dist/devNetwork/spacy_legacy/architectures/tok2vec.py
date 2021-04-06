from typing import List
from thinc.api import Model, chain, with_array, clone, residual, expand_window
from thinc.api import Maxout, Mish
from thinc.types import Floats2d
from spacy.tokens import Doc


def Tok2Vec_v1(
    embed: Model[List[Doc], List[Floats2d]],
    encode: Model[List[Floats2d], List[Floats2d]],
) -> Model[List[Doc], List[Floats2d]]:
    """Construct a tok2vec model out of embedding and encoding subnetworks.
    See https://explosion.ai/blog/deep-learning-formula-nlp

    embed (Model[List[Doc], List[Floats2d]]): Embed tokens into context-independent
        word vector representations.
    encode (Model[List[Floats2d], List[Floats2d]]): Encode context into the
        embeddings, using an architecture such as a CNN, BiLSTM or transformer.
    """
    receptive_field = encode.attrs.get("receptive_field", 0)
    tok2vec = chain(embed, with_array(encode, pad=receptive_field))
    tok2vec.set_dim("nO", encode.get_dim("nO"))
    tok2vec.set_ref("embed", embed)
    tok2vec.set_ref("encode", encode)
    return tok2vec


def MaxoutWindowEncoder_v1(
    width: int, window_size: int, maxout_pieces: int, depth: int
) -> Model[List[Floats2d], List[Floats2d]]:
    """Encode context using convolutions with maxout activation, layer
    normalization and residual connections.

    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    window_size (int): The number of words to concatenate around each token
        to construct the convolution. Recommended value is 1.
    maxout_pieces (int): The number of maxout pieces to use. Recommended
        values are 2 or 3.
    depth (int): The number of convolutional layers. Recommended value is 4.
    """
    cnn = chain(
        expand_window(window_size=window_size),
        Maxout(
            nO=width,
            nI=width * ((window_size * 2) + 1),
            nP=maxout_pieces,
            dropout=0.0,
            normalize=True,
        ),
    )
    model = clone(residual(cnn), depth)
    model.set_dim("nO", width)
    model.attrs["receptive_field"] = window_size * depth
    return model


def MishWindowEncoder_v1(
    width: int, window_size: int, depth: int
) -> Model[List[Floats2d], List[Floats2d]]:
    """Encode context using convolutions with mish activation, layer
    normalization and residual connections.

    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    window_size (int): The number of words to concatenate around each token
        to construct the convolution. Recommended value is 1.
    depth (int): The number of convolutional layers. Recommended value is 4.
    """
    cnn = chain(
        expand_window(window_size=window_size),
        Mish(nO=width, nI=width * ((window_size * 2) + 1), dropout=0.0, normalize=True),
    )
    model = clone(residual(cnn), depth)
    model.set_dim("nO", width)
    return model

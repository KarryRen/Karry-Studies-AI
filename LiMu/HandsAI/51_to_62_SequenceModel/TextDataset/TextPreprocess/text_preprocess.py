# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 15:19
# @Author  : Karry Ren

""" Pre-process the text data (`timemachine.txt` should be downloaded by download_data.py).
Transfer all text data to number.

"""

import re
from typing import List
import collections


def read_time_machine(txt_path: str = "/Users/karry/KarryRen/Codes/Karry-Studies-AI/LiMu"
                                      "/HandsAI/51_to_62_SequenceModel/TextDataset/TextPreprocess"
                                      "/timemachine.txt") -> list:
    """ Read the time machine text data.

    :param txt_path: the path of `.txt`data.

    return: adj_lines
        - a list, each item is one line in txt file
        - only have 27 types of characters, a~z and " "

    """

    # ---- Step 1. Read data from the txt file ---- #
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # ---- Step 2. Adjust the data ---- #
    adj_lines = [re.sub('[^A-Za-z]+', " ", line).strip().lower() for line in lines]

    return adj_lines


def tokenize(lines: List[str], token_type: str = "word") -> List[list]:
    """ Tokenize the lines, which means split the sentences in lines to token.

    :param lines: the list of sentences
    :param token_type: the target of sentences, you have 2 choices:
        - "word" : split the sentences to words
        - "char" : split the sentences to chars

    return: a 2D tokenized list of lines
        - each sentence will be a list
        - the item of each list of sentence is token_type

    """

    if token_type == "word":
        return [line.split() for line in lines]
    elif token_type == "char":
        return [list(line) for line in lines]
    else:
        raise TypeError(token_type)


class Vocab:
    """ The vocabulary of text. """

    def __init__(self, tokens: list = None, min_freq: int = 0, reserved_tokens=None):
        """ Init the vocab.

        :param tokens: a 1D or 2D list, each item is a list of tokens (2D) or just a list of token (1D)
            The item of tokens can be every thing ! Such as word, char or word tuple.
        :param min_freq: the minimum frequency, if the frequency of token < min_freq
            the token will be erased ! Make data easier for modeling.
        :param reserved_tokens: the reserved tokens which might not appear in tokens
            but will be added to the vocab.

        """

        # ---- Step 0. Build up the tokens ---- #
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # ---- Step 1. Count the tokens and sort by frequency from high to low ---- #
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # ---- Step 2. Add the unknown token and define idx and token transferring list / dict ---- #
        # idx_to_token and token_to_idx are y-1 relationship !
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # ---- Step 3. Construct the `token_to_idx` and `idx_to_token` dict (sorted by appearing frequency) ---- #
        for token, freq in self._token_freqs:
            # high-frequency token will have small idx and low-frequency token will have big idx.
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """ Use [token] to get the idx of token. """

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """ Use .to_tokens(idx) to get the token of idx. """

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens) -> collections.Counter:
    """ Count the token num in tokens.

    :param tokens: a 1D or 2D list of tokens

    return: the count result of all tokens

    """

    # Flatten the tokens
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(token_type: str = "char", max_tokens: int = -1) -> tuple:
    """ Load the pre-processed `time_machine` text data.

    :param token_type: the type of token ("char" or "word")
    :param max_tokens: the max token

    return:
        - corpus: the list of idx of each item in `time_machine` txt file
        - vocab: the vocab of time_machine

    """

    # ---- Load the `txt` data ---- #
    lines = read_time_machine()
    # ---- Tokenize ---- #
    tokens = tokenize(lines, token_type)
    # ---- Get the vocab ---- #
    vocab = Vocab(tokens)
    # ---- Get the idx of each item in lines ---- #
    corpus = [vocab[token] for line in tokens for token in line]
    # ---- Slice the max_token and return ---- #
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    corpus, vocab = load_corpus_time_machine(token_type="char")
    # print(vocab.to_tokens(corpus))

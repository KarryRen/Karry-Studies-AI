# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 16:58
# @Author  : Karry Ren

""" The dataset for text data. """

import random
import torch
from .TextPreprocess.text_preprocess import load_corpus_time_machine


def seq_data_iter_random(corpus: list, batch_size: int, num_steps: int):
    """ A random data iter for sequence data.
    Different from usual iter for `sin` data_iter, this iter will only use one data for one time.

    For Example if the length of dataset is 40, the num_steps is 4:
        - when we use the usual data_iter, there will be 36 items.
        - when we use this iter, there will be 10 or 9 items, no data will repeat !
            (the start idx is random for get random data, different epoch will read different data)
            (different iter data is random)

    :param corpus: the corpus of text data
    :param batch_size: the batch size
    :param num_steps: the num of steps

    """

    # ---- Slice the corpus randomly ---- #
    corpus = corpus[random.randint(0, num_steps - 1):]
    # ---- Compute the num of sequence and start idx ---- #
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # ---- Shuffle the start idx ---- #
    random.shuffle(initial_indices)

    def data(pos):
        """ Get the data from pos to pos + num_steps, totally num_steps data. """
        return corpus[pos: pos + num_steps]

    # ---- For loop the batch to yield batch data ---- #
    num_batches = num_subseqs // batch_size  # drop last
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]  # shape=(batch_size, )
        X = [data(j) for j in initial_indices_per_batch]  # shape=(batch_size, )
        Y = [data(j + 1) for j in initial_indices_per_batch]  # shape=(batch_size, )
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus: list, batch_size: int, num_steps: int):
    """ A sequential data iter for sequence data.
    Different from usual iter for `sin` data_iter, this iter will only use one data for one time.

    For Example if the length of dataset is 40, the num_steps is 4:
        - when we use the usual data_iter, there will be 36 items.
        - when we use this iter, there will be 10 or 9 items, no data will repeat !
            (the start idx is random for get random data, different epoch will read different data)
            (different iter data is sequential)

    :param corpus: the corpus of text data
    :param batch_size: the batch size
    :param num_steps: the num of steps

    """

    # ---- Compute the start_offset ---- #
    offset = random.randint(0, num_steps)

    # ---- Drop last to compute num of tokens ---- #
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    # ---- Get the x and y, shape=(num_tokens,) ---- #
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    # ---- Reshape and Load shape=(bs, num_tokens/bs)---- #
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """ The dataloader for sequence data. """

    def __init__(self, batch_size: int, num_steps: int, use_random_iter: bool = False, max_tokens: int = -1):
        """ Init of Dataloader.

        :param batch_size: the batch size
        :param num_steps: the num of steps
        :param use_random_iter: use the random iter or not
        :param max_tokens: the max token

        """

        # ---- Get the seq iter ---- #
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential

        # ---- Load the time machine corpus ---- #
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens=max_tokens)

        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


if __name__ == "__main__":  # A demo
    my_seq = list(range(35))
    print("random:")
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print("X: ", X, "\nY:", Y)
    print("sequential")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print("X: ", X, "\nY:", Y)
    print("test seq dataloader")

    a = SeqDataLoader(batch_size=2, num_steps=2, use_random_iter=False)
    for x, y in a:
        print(x, y)
        break

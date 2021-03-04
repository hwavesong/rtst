# -*- coding: utf-8 -*-
import logging
import random
import sys
from itertools import product

sys.path.insert(0, '..')

import numpy as np
import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def produce_data_from_word_bucket_pair(word_bucket1, word_bucket2, filter_table, raw_wv):
    raw_AB_vectors, raw_AC_vectors, raw_A = list(), list(), list()

    for raw_AB_word, raw_AC_word in product(word_bucket1, word_bucket2):
        # filter word pair in word similarity test tasks
        if raw_AB_word not in raw_wv or raw_AC_word not in raw_wv: continue
        if (raw_AB_word, raw_AC_word) in filter_table or (raw_AC_word, raw_AB_word) in filter_table: continue

        raw_AB_vectors.append(raw_wv[raw_AB_word])
        raw_AC_vectors.append(raw_wv[raw_AC_word])

    raw_AB_vectors = np.array(raw_AB_vectors)
    raw_AC_vectors = np.array(raw_AC_vectors)

    raw_A = utils.calculate_cosine_score(raw_AB_vectors, raw_AC_vectors)

    np.clip(raw_AB_vectors, -1., 1., raw_AB_vectors)
    np.clip(raw_AC_vectors, -1., 1., raw_AC_vectors)

    return raw_AB_vectors.tolist(), raw_AC_vectors.tolist(), raw_A.tolist()


def generate_data(bucket_data, bucket_pairs, filter_table, raw_wv, cfg):
    epochs = cfg.getint('epochs')
    batch_size = cfg.getint('batch_size')

    shuffle_index = create_shuffle_index()

    batch_raw_AB = list()
    batch_raw_AC = list()
    batch_A = list()

    for _ in range(epochs):
        for b1, b2, _ in bucket_pairs:
            raw_AB, raw_AC, raw_A = produce_data_from_word_bucket_pair(bucket_data[b1], bucket_data[b2], filter_table, raw_wv)

            batch_raw_AB.extend(raw_AB)
            batch_raw_AC.extend(raw_AC)
            batch_A.extend(raw_A)

            batch_len = len(batch_raw_AB)

            if batch_len < 10240: continue

            batch_raw_AB = np.array(batch_raw_AB, np.float32)
            batch_raw_AC = np.array(batch_raw_AC, np.float32)
            batch_A = np.array(batch_A, np.float32)

            batch_raw_AB[:10240] = batch_raw_AB[:10240][shuffle_index]
            batch_raw_AC[:10240] = batch_raw_AC[:10240][shuffle_index]
            batch_A[:10240] = batch_A[:10240][shuffle_index]

            for i in range(10240 // batch_size):
                start, end = i * batch_size, (i + 1) * batch_size
                yield batch_raw_AB[start:end], batch_raw_AC[start:end], batch_A[start:end]

            # quotient, remainder = divmod(b_len, batch_size)
            remainder = batch_len - 10240
            if remainder == 0:  # if remiander==0, return a new list
                remainder = -batch_len

            batch_raw_AB = batch_raw_AB[-remainder:].tolist()
            batch_raw_AC = batch_raw_AC[-remainder:].tolist()
            batch_A = batch_A[-remainder:].tolist()


def create_shuffle_index():
    random.seed(37)

    shuffle_index = [i for i in range(10240)]
    random.shuffle(shuffle_index)

    shuffle_index = np.array(shuffle_index)

    return shuffle_index


def generate_evaluate_data(bucket_data, bucket_pairs, filter_table, glove_wv):
    for b1, b2, _ in bucket_pairs:
        yield produce_data_from_word_bucket_pair(bucket_data[b1], bucket_data[b2], filter_table, glove_wv)

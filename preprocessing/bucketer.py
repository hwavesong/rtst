# -*- coding: utf-8 -*-
import logging
import multiprocessing
import sys
import time
from itertools import product

sys.path.insert(0, '..')

import nltk
import numpy as np

import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def bucket_word(cfg):
    logging.info('step1: bucket word')
    source_path = cfg['file_path']
    target_path = cfg['bucket_path']

    words = get_words(source_path)

    partOfSpeech_to_words = tag_PartOfSpeech(words)

    partOfSpeech_to_buckets = split_buckets_in_partOfSpeech(partOfSpeech_to_words)

    write_partOfSpeech_buckets_to_file(partOfSpeech_to_buckets, target_path)

    logging.info('done')


def write_partOfSpeech_buckets_to_file(merge_pos_cnt, target_path):
    logging.info('writing to ' + target_path)

    with open(target_path, 'w', encoding='utf8') as file:
        for partOfSpeech, word_buckets in merge_pos_cnt.items():
            file.write('\n')

            for word_bucket in word_buckets:
                file.write('\n')
                for word in word_bucket:
                    file.write(word + '\n')


def split_buckets_in_partOfSpeech(partOfSpeech_to_words):
    partOfSpeech_to_buckets = dict()
    remaining_words = list()

    for partOfSpeech, words in partOfSpeech_to_words.items():
        if len(words) < 32:  # collect all small number words
            remaining_words.extend(words)
        else:
            quotient, remainder = divmod(len(words), 32)

            if quotient > 0:
                partOfSpeech_to_buckets[partOfSpeech] = [words[m * 32:(m + 1) * 32] for m in range(quotient)]

            if remainder > 0:
                remaining_words.extend(words[-remainder:])

    partOfSpeech_to_buckets['fused'] = split_remainingWords_buckets(remaining_words)

    return partOfSpeech_to_buckets


def split_remainingWords_buckets(remaining_words):
    remaining_words_quotient, remaining_words_remainder = divmod(len(remaining_words), 32)

    last_value = []

    if remaining_words_quotient > 0:
        last_value = [remaining_words[m * 32:(m + 1) * 32] for m in range(remaining_words_quotient)]

    if remaining_words_remainder > 0:
        last_value.append(remaining_words[-remaining_words_remainder:])

    return last_value


def tag_PartOfSpeech(words):
    logging.info('Dividing buckets by the part of speech tagger of NLTK')

    pos_word_dct = dict()
    # nltk.download('averaged_perceptron_tagger')
    for w, p in nltk.pos_tag(words):
        if p not in pos_word_dct.keys():
            pos_word_dct[p] = list(w)
            continue

        pos_word_dct[p].append(w)

    return pos_word_dct


def get_words(source_path):
    wv_dct = utils.load_wv_dct(source_path)

    words = list(wv_dct.keys())

    return words


def sample_buckets(cfg):
    logging.info('step2: sample buckets')

    bucket_path = cfg['bucket_path']
    sample_rate = int(cfg['sample_rate'])
    saved_path = cfg['sampled_path']

    bucketed_words = utils.load_bucket(bucket_path)

    sampled_bucket = _sample_buckets(bucketed_words, sample_rate)

    write_bucket_to_file(sampled_bucket, saved_path)


def write_bucket_to_file(sampled_bucket, saved_path):
    with open(saved_path, 'w', encoding='utf8') as file:
        for one_bucket in sampled_bucket:
            file.write('\n')

            for b in one_bucket:

                file.write('\n')
                for w in b:
                    file.write(w + '\n')


def _sample_buckets(bucketed_words, sample_rate):
    sample_bucket = list()
    for one_bucket in bucketed_words:
        bucket_num = len(one_bucket)

        if bucket_num < sample_rate:
            sample_bucket.append(one_bucket)
        else:
            sample_list = list(bucket for ix, bucket in enumerate(one_bucket) if ix % sample_rate == 0)
            sample_bucket.append(sample_list)
    return sample_bucket


def get_cos_sim(m1, m2, outter_ix, inner_ix):
    return outter_ix, inner_ix, np.mean(utils.calculate_cosine_score(m1, m2))


def calculate_word_bucket_pair_similarity_score(cfg):
    logging.info('computing similarity bucket pairs')

    wv_dct = utils.load_wv_dct(cfg['file_path'])
    filter_table = utils.load_filter_set(cfg['filter_table_path'])
    sample_data = utils.load_bucket(cfg['sampled_path'])
    flatten_list = utils.flatten_bucket(sample_data)

    saved_path = cfg['bucket_pairs_path']

    pairs_sim = list()
    packed_data = list()

    cpu_num = get_cpu_num()
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        for outter_ix, outter_bucket in enumerate(flatten_list):
            spent_time = (time.time() - start_time) // 60
            logging.info('%.1f => 100 , spent time : %d min' % ((outter_ix * 100 / len(flatten_list)), spent_time))

            for inner_ix, inner_bucket in enumerate(flatten_list):
                if inner_ix <= outter_ix: continue

                if len(packed_data) >= cpu_num * 64:
                    pairs_sim.extend(pool.starmap(get_cos_sim, packed_data))

                    del packed_data
                    packed_data = list()

                v1s, v2s = list(), list()
                for x1, x2 in product(outter_bucket, inner_bucket):
                    if x1 not in wv_dct.keys() or x2 not in wv_dct.keys():
                        continue
                    if (x1, x2) in filter_table or (x2, x1) in filter_table:  # letter insensitive
                        continue

                    v1, v2 = wv_dct[x1], wv_dct[x2]

                    v1s.append(v1)
                    v2s.append(v2)

                v1_matrix, v2_matrix = np.array(v1s), np.array(v2s)

                packed_data.append((v1_matrix, v2_matrix, outter_ix, inner_ix))

        pairs_sim.extend(pool.starmap(get_cos_sim, packed_data))
        pool.close()
        pool.join()

    with open(saved_path, 'w', encoding='utf8') as file:
        for one_piece in pairs_sim:
            file.write('\t'.join([str(x) for x in one_piece]) + '\n')

    logging.info('done')


def get_cpu_num():
    return multiprocessing.cpu_count()


def sort_bucket_pairs(cfg):
    sim_bucket_pairs_path = cfg['bucket_pairs_path']
    saved_path = cfg['sorted_bucket_pairs_path']

    bucket_pairs = utils.load_bucket_pairs(sim_bucket_pairs_path)

    bucket_pairs.sort(key=lambda x: abs(x[-1]))

    with open(saved_path, 'w', encoding='utf8') as target_file:
        for line in bucket_pairs:
            target_file.write('\t'.join([str(x) for x in line]) + '\n')


if __name__ == '__main__':
    # load config file 
    cfg = utils.read_config('./setup.ini', 'glove')

    # step1 : bucket words by part of speech.
    bucket_word(cfg)

    # step2 : sample data, defalut rate is 50
    sample_buckets(cfg)

    # step3 : get sim bucket pairs
    multiprocessing.set_start_method('forkserver')
    calculate_word_bucket_pair_similarity_score(cfg)

    # step4 : thershold select
    sort_bucket_pairs(cfg)

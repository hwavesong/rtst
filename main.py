# -*- coding: utf-8 -*-
import os
import time
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from absl import app, logging

import compressor
import model as cm
import utils
from data_builder import data_factory


def train_model(train_dataset, model_saved_path):
    model = cm.CompressModel()

    optimizer = cm.get_train_op()

    last_16_loss = deque([1.] * 16, maxlen=16)

    start_time = time.time()

    for ix, (raw_AB, raw_AC, raw_A) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            short_AB, short_AC = model(raw_AB), model(raw_AC)

            A_hat = cm.calculate_angle(short_AB, short_AC)

            RTST_A_loss = cm.calculate_RTST_A_loss(raw_A, A_hat)
            RTST_BC_loss = cm.calculate_RTST_BC_loss(raw_AB, raw_AC, short_AB, short_AC)

            Bin_loss = cm.calculate_Bin_loss(short_AB, short_AC)  # max

            total_loss = tf.math.log(RTST_A_loss + RTST_BC_loss) - tf.math.log(Bin_loss)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        last_16_loss.append(total_loss)

        if ix % 1000 == 0:
            spent_time = (time.time() - start_time) // 60
            mean_loss = np.mean(last_16_loss)
            logging.info('{: >5,d} => total_loss : {:.4f} : spent : {: >2d} min : mean loss : {:.4f}'.format(ix, total_loss, int(spent_time), mean_loss))

    model.save_weights(model_saved_path, save_format='tf')

    logging.info('training phase done')


def main(argv):
    del argv

    cfg = utils.read_config('setup.ini', 'glove_128')

    word_to_vectors = utils.load_wv_dct(cfg['file_path'])

    filter_table = utils.load_filter_set(cfg['filter_path'])

    bucket_data = utils.load_bucket(cfg['sample_path'])
    bucket_data = utils.flatten_bucket(bucket_data)

    bucket_pairs = utils.load_bucket_pairs(cfg['sorted_bucket_pairs_path'])

    train_bucket_pairs = bucket_pairs[:2000]

    # Train
    training_dataset = data_factory.generate_data(bucket_data, train_bucket_pairs, filter_table, word_to_vectors, cfg)
    train_model(training_dataset, cfg['compress_model_path'])

    # Gernerate compress vector and binary vector
    select_t_bucket_pairs = bucket_pairs[2000:2050]

    grid_search_dataset = data_factory.generate_evaluate_data(bucket_data, select_t_bucket_pairs, filter_table, word_to_vectors)
    threshold = compressor.solve_truncate_threshold(grid_search_dataset, cfg)

    # Dimensionality reduction
    compressor.produce_short_vectors(word_to_vectors, cfg)

    # Binarization
    compressor.produce_binary_vectors(threshold, cfg)


if __name__ == '__main__':
    app.run(main)

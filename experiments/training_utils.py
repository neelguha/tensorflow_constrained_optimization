# A list of training utility functions 
import sys
sys.path.insert(0,'/Users/neelguha/Dropbox/NeelResearch/fairness/code/tensorflow_constrained_optimization/')
import math
import random
import numpy as np
import pandas as pd
import warnings
from six.moves import xrange
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt
import logging 
import time

def training_generator(model, train_df, test_df, minibatch_size, num_iterations_per_loop=1, num_loops=1):
    random.seed(31337)
    num_rows = train_df.shape[0]
    minibatch_size = min(minibatch_size, num_rows)
    permutation = list(range(train_df.shape[0]))
    random.shuffle(permutation)

    session = tf.Session()
    session.run((tf.global_variables_initializer(),
               tf.local_variables_initializer()))

    minibatch_start_index = 0
    for n in xrange(num_loops):
        for _ in xrange(num_iterations_per_loop):
            minibatch_indices = []
            while len(minibatch_indices) < minibatch_size:
                minibatch_end_index = (
                minibatch_start_index + minibatch_size - len(minibatch_indices))
                if minibatch_end_index >= num_rows:
                    minibatch_indices += range(minibatch_start_index, num_rows)
                    minibatch_start_index = 0
                else:
                    minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                    minibatch_start_index = minibatch_end_index
                    
            session.run(
                  model.train_op,
                  feed_dict=model.feed_dict_helper(
                      train_df.iloc[[permutation[ii] for ii in minibatch_indices]]))

        train_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
        session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
        session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(train_df))
        test_predictions = session.run(
            model.predictions_tensor,
            feed_dict=model.feed_dict_helper(test_df))
        yield (train_predictions, test_predictions)

def error_rate(predictions, labels):
    signed_labels = (
      (labels > 0).astype(np.float32) - (labels <= 0).astype(np.float32))
    numerator = (np.multiply(signed_labels, predictions) <= 0).sum()
    denominator = predictions.shape[0]
    return float(numerator) / float(denominator)


def positive_prediction_rate(predictions, subset):
    numerator = np.multiply((predictions > 0).astype(np.float32),
                          (subset > 0).astype(np.float32)).sum()
    denominator = (subset > 0).sum()
    return float(numerator) / float(denominator)

def tpr(df, label_column):
    """Measure the true positive rate."""
    fp = sum((df['predictions'] >= 0.0) & (df[label_column] > 0.5))
    ln = sum(df[label_column] > 0.5)
    return float(fp) / float(ln)

def _get_error_rate_and_constraints(df, tpr_max_diff, label_column, protected_columns):
    """Computes the error and fairness violations."""
    error_rate_local = error_rate(df[['predictions']], df[[label_column]])
    overall_tpr = tpr(df, label_column)
    diffs = []
    for protected_attribute in protected_columns:
        diffs.append((overall_tpr - tpr_max_diff) - tpr(df[df[protected_attribute] > 0.5], label_column))
    return error_rate_local, diffs

def _get_exp_error_rate_constraints(cand_dist, error_rates_vector, constraints_matrix):
    """Computes the expected error and fairness violations on a randomized solution."""
    expected_error_rate = np.dot(cand_dist, error_rates_vector)
    expected_constraints = np.matmul(cand_dist, constraints_matrix)
    return expected_error_rate, expected_constraints

def training_helper(model,
                    train_df,
                    test_df,
                    minibatch_size,
                    label_column,
                    protected_columns,
                    num_iterations_per_loop=1,
                    num_loops=1 ,
                    interval = 5):
    train_error_rate_vector = []
    train_constraints_matrix = []
    test_error_rate_vector = []
    test_constraints_matrix = []
    iteration = 1
    start = time.time()
    for train, test  in training_generator(
      model, train_df, test_df, minibatch_size, num_iterations_per_loop,
      num_loops):
        train_df['predictions'] = train
        test_df['predictions'] = test
        if (iteration - 1) % interval == 0:
            train_error_rate, train_constraints = _get_error_rate_and_constraints(
            train_df, model.tpr_max_diff, label_column, protected_columns)
            train_error_rate_vector.append(train_error_rate)
            train_constraints_matrix.append(train_constraints)

            test_error_rate, test_constraints = _get_error_rate_and_constraints(
                test_df, model.tpr_max_diff, label_column, protected_columns)
            test_error_rate_vector.append(test_error_rate)
            test_constraints_matrix.append(test_constraints)
            duration = time.time() - start
            logging.info(
                "Finished %d/%d. Train error = %f. Max train violation = %f. Test error = %f. Max test violation = %f. %f seconds" % 
                (iteration, num_loops, train_error_rate, max(train_constraints), test_error_rate, max(test_constraints), duration)
            )
        else:
            duration = time.time() - start
            logging.info(
                "Finished %d/%d.  %f seconds" % 
                (iteration, num_loops, duration)
            )
        iteration += 1
        start = time.time()
    return (train_error_rate_vector, train_constraints_matrix, test_error_rate_vector, test_constraints_matrix)

def get_tpr_subset(df, subsets, label_column):
    filtered = df 
    for subset in subsets:
        filtered = filtered[filtered[subset] > 0]
    return tpr(filtered, label_column)

def get_acc_subset(df, subsets):
    filtered = df 
    for subset in subsets:
        filtered = filtered[filtered[subset] > 0]
    predictions = filtered['predictions']
    labels = filtered['label']
    return np.mean(np.array(predictions > 0.0) == np.array(labels > 0.0))
    
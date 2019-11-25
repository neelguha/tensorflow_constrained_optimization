import logging
from models import *
from training_utils import *
from data_utils import *


def run_eo_experiment(
    train_df, test_df, feature_names, label_column, 
    active_protected_columns, all_protected_columns, epochs=40, 
    minibatch_size=32,  max_diff=0.05, lr=0.001):
    ''' Runs equalized odds fairness experiment. 

    Args:
        train_df: train dataframe
        test_df: test dataframe 
        feature_names: names of features 
        label_column: name of column corresponding to label 
        active_protected_columns: names of columns on which we place constraints
        all_protected_columns: all columns considered proctected
        epochs: number of training epochs
        minibatch_size: size of minibatches
        max_diff: maximum tolerated difference for TPR 
        lr: learning rate


    Returns:
        constraint_violations_train: dictionary of constraint violations by protected group over the training data 
        constraint_violations_test: dictionary of constraint violations by protected group over test data 
        performance_test: dictionary of performance by each protected group over test_data
    '''

    num_iterations_per_loop = int(len(train_df) / minibatch_size)
    if len(active_protected_columns) == 0:
        # There are no protected columns, so we just run the baseline
        logging.info("Running baseline.")
        logging.info("Minibatch size: %d. Epochs: %d. Maxx diff: %f" %
                     (minibatch_size, epochs, max_diff))
        model = LinearModel(feature_names, active_protected_columns,
                            label_column, None, tpr_max_diff=max_diff)
        model.build_train_op(max_diff, unconstrained=True)

        # training_helper returns the list of errors and violations over each epoch.
        train_errors, train_violations, test_errors, test_violations = training_helper(
            model, train_df, test_df, minibatch_size, label_column, active_protected_columns, 
            num_iterations_per_loop=num_iterations_per_loop, num_loops=epochs
        )
        
    else:
        logging.info("Running Model!")
        constraints = [[p] for p in active_protected_columns]
        model = LinearModel(feature_names, active_protected_columns,
                            label_column, constraints, tpr_max_diff=max_diff)
        model.build_train_op(0.01, unconstrained=False)
        logging.info("Training!")
        # training_helper returns the list of errors and violations over each epoch.
        train_errors, train_violations, test_errors, test_violations = training_helper(
            model, train_df, test_df, minibatch_size, label_column, active_protected_columns, num_iterations_per_loop=num_iterations_per_loop, num_loops=epochs
        )

    train_out = create_result_to_save(
        train_df, all_protected_columns, label_column)
    test_out = create_result_to_save(
        test_df, all_protected_columns, label_column)

    train_violation = score_violations_df(train_out, max_diff)
    test_violation = score_violations_df(test_out, max_diff)
    scores = score_results(test_out)

    return train_violation, test_violation, scores



def get_group_tpr_rates(df):
    overall_tpr_rate = tpr(df, "label")
    columns = list(df.keys())
    col_rates = {}
    for col in columns: 
        if col in ['label', 'predictions', 'predicted_class']:
            continue 
        col_rates[col] = tpr(df[df[col] == 1], "label")
    return col_rates


def score_violations_df(df, max_diff):
    ''' Generate constraint violation scores. 

    Args:
        df: dataframe with label column
        max_diff
    '''

    overall = tpr(df, "label")
    group_rates = get_group_tpr_rates(df)
    violations = {}
    for key, val in group_rates.items():
        violations[key] = max(max_diff - (overall - val), 0)
    return violations

def score_results(df):

    # get tpr rate
    tpr_rate = tpr(df, 'label')

    # get accuracy 
    accuracy = get_accuracy(df)
    return {
        'tpr': tpr_rate,
        'accuracy': accuracy
    }

def get_accuracy(df):
    return np.mean(df['predicted_class'] == df['label'])
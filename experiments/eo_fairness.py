# Experiments for equalized odds fairness on the Adult Income dataset

from models import *
from training_utils import *
from data_utils import *
import argparse, logging, json
from termcolor import colored
import tensorflow as tf 


format_out = colored('[%(asctime)s]', 'blue') + ' %(message)s'
logging.basicConfig(format=format_out,
                    datefmt='%m/%d/%Y %I:%M:%S%p', level=logging.INFO)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="adult-income",
                    help='name of dataset')
parser.add_argument('--baseline', action='store_true',
                    help="Run (unfair) baseline.")
parser.add_argument('--super', action='store_true',
                    help="Apply constraints at the super group level")
parser.add_argument('--sub', action='store_true',
                    help="Apply constraints at the sub group level")
parser.add_argument('--max_diff', default=0.05,
                    help="Maximum difference for TPR")
parser.add_argument('--race', action='store_true',
                    help="Whether to treat race as a protected attribute")
parser.add_argument('--age', action='store_true',
                    help="Whether to treat age as a protected attribute")
parser.add_argument('--gender', action='store_true',
                    help="Whether to treat gender as a protected attribute")
parser.add_argument('--exp_name', help = "Name of experiment - used for saving results.")
parser.add_argument('--gpu', action = 'store_true', help = "Whether to use GPU")
args = parser.parse_args()


def main():

    logging.info("Running equalized odds fairness via rate constraints...")
    logging.info("Dataset: %s." % (args.dataset))

    protected_prefixes = []
    if args.race:
        protected_prefixes.append('race')
        logging.info("Adding race as protected attribute..")
    if args.gender:
        protected_prefixes.append('gender')
        logging.info("Adding gender as protected attribute..")
    if args.age:
        protected_prefixes.append('age')
        logging.info("Adding age as protected attribute..")

    # get data
    logging.info("loading data...")
    if args.dataset == 'adult-income':
        train_df, test_df, feature_names, protected_columns, label_column = load_adult_data(
            protected_prefixes)
    elif args.dataset == 'ipums':
        train_df, test_df, feature_names, protected_columns, label_column = get_ipums_income(protected_prefixes)
    elif args.dataset == 'ipums-small':
        train_df, test_df, feature_names, protected_columns, label_column = get_ipums_income(protected_prefixes, small=True)
    else:
        raise Exception("Uknown dataset: %s" % args.dataset)

    logging.info("Loaded Data.")
    assert(len(protected_columns) > 0)
    logging.info("%d training samples. %d test samples. %d features. %d protected attributes." %
                 (len(train_df), len(test_df), train_df.shape[1], len(protected_columns)))
    logging.info(protected_columns)
    epochs = 40
    minibatch_size = 32
    num_iterations_per_loop = int(len(train_df) / minibatch_size)
    if args.baseline:
        logging.info("Running baseline.")
        logging.info("Minibatch size: %d. Epochs: %d" %
                     (minibatch_size, epochs))
        model = LinearModel(feature_names, protected_columns,
                            label_column, None, tpr_max_diff=0.05)
        model.build_train_op(args.max_diff, unconstrained=True)

        # training_helper returns the list of errors and violations over each epoch.
        train_errors, train_violations, test_errors, test_violations = training_helper(
            model, train_df, test_df, minibatch_size, label_column, protected_columns, num_iterations_per_loop=num_iterations_per_loop, num_loops=epochs
        )
        train_out = create_result_to_save(train_df, protected_columns, label_column)
        test_out = create_result_to_save(test_df, protected_columns, label_column)
        parameters = {
            'epochs': epochs, 
            'minibatch_size': minibatch_size,
            'model_selection': "last",
            'type': 'baseline'
        }
    else:
        print("Running Model!")
        constraints = [[p] for p in protected_columns]
        model = LinearModel(feature_names, protected_columns,
                            label_column, constraints, tpr_max_diff=args.max_diff)
        model.build_train_op(0.01, unconstrained=False)
        print("Training!")
        # training_helper returns the list of errors and violations over each epoch.
        train_errors, train_violations, test_errors, test_violations = training_helper(
            model, train_df, test_df, 100, label_column, protected_columns, num_iterations_per_loop=num_iterations_per_loop, num_loops=epochs
        )
        train_out = create_result_to_save(train_df, protected_columns, label_column)
        test_out = create_result_to_save(test_df, protected_columns, label_column)
        parameters = {
            'epochs': epochs, 
            'minibatch_size': minibatch_size,
            'model_selection': "last",
            'type': 'tpr',
            'tpr_diff': args.max_diff
        }


    # save to output directory 
    output_directory = os.path.join(RESULTS_DIR, 'rate_constraints', args.dataset, args.exp_name)
    logging.info("Saving all results to directory %s" % output_directory)
    make_dir(output_directory)
    logging.info("Saving train output...")
    train_out.to_pickle(os.path.join(output_directory, 'train_out'))
    logging.info("Saving test output...")
    test_out.to_pickle(os.path.join(output_directory, 'test_out'))
    logging.info("Saving experiment parameters...")
    with open(os.path.join(output_directory, 'parameters.json'), 'w') as out_file: 
        json.dump(parameters, out_file)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()

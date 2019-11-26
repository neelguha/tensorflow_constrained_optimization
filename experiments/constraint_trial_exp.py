# See https://app.gitbook.com/@neelguha/s/fairness/~/drafts/-LtzyZhuwXbrY56TclXy/adult-income-dataset for explanation. 

import os, sys, json
from models import *
from training_utils import *
from data_utils import *
from run_fairness import *
from termcolor import colored
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="adult-income",
                    help='name of dataset')
args = parser.parse_args()

format_out = colored('[%(asctime)s]', 'blue') + ' %(message)s'
logging.basicConfig(format=format_out,
                    datefmt='%m/%d/%Y %I:%M:%S%p', level=logging.INFO)

K = 1
def main():
    logging.info("Dataset: {}".format(args.dataset))

    if args.dataset == 'adult':
        train_df, test_df, feature_names, label_column = load_adult_data()
        all_protected_columns = get_protected_attributes(args.dataset, feature_names)
        NUM_PROTECTED = [0, 5, 10, 15, 22]
    elif args.dataset == 'ipums-small':
        train_df, test_df, feature_names, label_column = get_ipums_income(small= True)
        all_protected_columns = get_protected_attributes(args.dataset, feature_names, label_column, train_df, test_df)
        NUM_PROTECTED = []
        intervals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for ratio in intervals:
            NUM_PROTECTED.append(int(ratio*len(all_protected_columns)))
    print("Training shape:", train_df.shape)    
    print("Test shape:", test_df.shape)
    logging.info("%d protected attributes" % len(all_protected_columns))
    return
    all_results = {}

    for num_protected in NUM_PROTECTED:
        print("="*10, "Num protected: %d" % num_protected, "="*10)
        trial_results = {}
        for k in range(K):
            print("="*5, "Trial %d" % k, "="*5)
            active_protected = np.random.choice(all_protected_columns, num_protected, replace = False)
            train_violation, train_rates, test_violation, test_rates, scores = run_eo_experiment( 
                train_df, test_df, feature_names, label_column, 
                active_protected, all_protected_columns, epochs=20, 
                minibatch_size=32,  max_diff=0.05, lr=0.005
            )
            trial_results[k] = {
                'train_violation': train_violation, 
                'train_rates': train_rates,
                'test_violation': test_violation,
                'test_rates': test_rates,
                'scores': scores,
                'active_constraints': active_protected.tolist()
            }
        all_results[num_protected] = trial_results
    
     # save to output directory 
    output_directory = os.path.join(RESULTS_DIR, 'rate_constraints', args.dataset, "trials")
    logging.info("Saving all results to directory %s" % output_directory)
    make_dir(output_directory)
    logging.info("Saving experiment output...")
    with open(os.path.join(output_directory, 'results.json'), 'w') as out_file:
        json.dump(all_results, out_file)
    





if __name__ == "__main__":
    main()
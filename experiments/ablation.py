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
def main():
    logging.info("Dataset: {}".format(args.dataset))

    if args.dataset == 'adult':
        train_df, test_df, feature_names, label_column = load_adult_data()
        all_protected_columns = get_protected_attributes(args.dataset, feature_names)
    elif args.dataset == 'ipums-small':
        train_df, test_df, feature_names, label_column = get_ipums_income(small= True)
        all_protected_columns = get_protected_attributes(args.dataset, feature_names, label_column, train_df, test_df)
    print("Training shape:", train_df.shape)    
    print("Test shape:", test_df.shape)
    logging.info("%d protected attributes" % len(all_protected_columns))
    
    all_results = {}

    for protected in all_protected_columns:
        print("="*10, "Protected: %s" % protected, "="*10)
        active_protected = [protected]
        train_violation, train_rates, test_violation, test_rates, scores = run_eo_experiment( 
            train_df, test_df, feature_names, label_column, 
            active_protected, all_protected_columns, epochs=10, 
            minibatch_size=128,  max_diff=0.05, lr=0.01
        )
        trial_results = {
            'train_violation': train_violation, 
            'train_rates': train_rates,
            'test_violation': test_violation,
            'test_rates': test_rates,
            'scores': scores,
            'active_constraints': protected
        }
        all_results[protected] = trial_results
    
        # save to output directory 
        output_directory = os.path.join(RESULTS_DIR, 'rate_constraints', args.dataset, "ablation")
        logging.info("Saving current results to directory %s" % output_directory)
        make_dir(output_directory)
        logging.info("Saving experiment output...")
        with open(os.path.join(output_directory, 'results.json'), 'w') as out_file:
            json.dump(all_results, out_file)
    





if __name__ == "__main__":
    main()
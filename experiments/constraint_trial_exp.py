# See https://app.gitbook.com/@neelguha/s/fairness/~/drafts/-LtzyZhuwXbrY56TclXy/adult-income-dataset for explanation. 

import os, sys, json
from models import *
from training_utils import *
from data_utils import *
from run_fairness import *
from termcolor import colored

format_out = colored('[%(asctime)s]', 'blue') + ' %(message)s'
logging.basicConfig(format=format_out,
                    datefmt='%m/%d/%Y %I:%M:%S%p', level=logging.INFO)

K = 1

NUM_PROTECTED = [0, 5, 10, 15, 22]

DATASET = 'adult-income'

def main():
    train_df, test_df, feature_names, label_column = load_adult_data()
    all_protected_columns = get_protected_attributes(DATASET, feature_names)
    
    logging.info("%d protected attributes" % len(all_protected_columns))
    all_results = {}

    for num_protected in NUM_PROTECTED:
        print("="*10, "Num protected: %d" % num_protected, "="*10)
        trial_results = {}
        for k in range(K):
            print("="*5, "Trial %d" % k, "="*5)
            active_protected = np.random.choice(all_protected_columns, num_protected, replace = False)
            train_violation, test_violation, scores = run_eo_experiment( 
                train_df, test_df, feature_names, label_column, 
                active_protected, all_protected_columns, epochs=20, 
                minibatch_size=32,  max_diff=0.05, lr=0.005
            )
            trial_results[k] = {
                'train_violation': train_violation.tolist(), 
                'test_violation': test_violation.tolist(),
                'scores': scores.tolist(),
                'active_constraints': active_protected.tolist()
            }
        all_results[num_protected] = trial_results
    
     # save to output directory 
    output_directory = os.path.join(RESULTS_DIR, 'rate_constraints', DATASET, "trials")
    logging.info("Saving all results to directory %s" % output_directory)
    make_dir(output_directory)
    logging.info("Saving experiment output...")
    with open(os.path.join(output_directory, 'results.json'), 'w') as out_file:
        json.dump(all_results, out_file)
    





if __name__ == "__main__":
    main()
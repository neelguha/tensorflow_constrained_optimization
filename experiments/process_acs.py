# code for creating ACS fairness dataset 

import os, sys 
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from collections import Counter

# parameters describing code book 
HEADER_END = 9
VARIABLE_LIST = [10, 92]
VAL_LIST = [99, 4167]


def in_range(R, l):
    return l >= R[0] and l <= R[1]


def load_codebook(data_dir):
    lines = ['DUMMY']
    with open(os.path.join(data_dir, "codebook.txt"), "r") as in_file:
        line_num = 0
        for line in in_file:
            lines.append(line.strip())

    var_cols = {} # the columns containing data for a variable
    var_descriptions = {} # the column descriptions for a variable 
    var_values = {} # a map of the numeric codes to the values they signify for different variables 

    line_num = 0
    active_var = None
    in_box = False
    while True:
        line_num += 1
        if line_num == len(lines):
            break
        line = lines[line_num]
            
        if line_num < HEADER_END:
            continue

        if in_range(VARIABLE_LIST, line_num): 
            items = line.strip().replace("\t", " ").split(" ")
            items = [x for x in items if not x == ""]
            var_name, _, cols, length, _ = items
            length = int(length)
            start_col = int(cols.split('-')[0])
            var_cols[var_name] = list(range(start_col, start_col+length))
        
        if  in_range(VAL_LIST, line_num):
            if len(line) == 0:
                in_box = False
                active_var = None
                continue
            
            line = line.replace("\t", " ")
            items = line.split(" ")
            if in_box:
                if len(items) <2:
                    continue
                code = items[0]
                
                # check that code is correct length 
                if len(code) < len(var_cols[active_var]):
                    while len(code) <  len(var_cols[active_var]):
                        code = '0' + code
                
                value = ' '.join(items[1:])
                var_values[active_var][code] = value
            else:
                # starting new var descriptor 
                var_name = items[0]
                var_desc = ' '.join(items[1:])
                var_descriptions[var_name] = var_desc
                print("Adding Variable (%s): %s" % (var_name, var_desc))
                active_var = var_name
                var_values[active_var] = {}
                in_box = True 
    return var_cols, var_descriptions, var_values

def load_raw(data_dir):
    data = []
    count = 0
    with open(os.path.join(data_dir, 'usa_00003.dat'), 'r') as in_file:
        for line in in_file:
            count += 1 
            txt = line.strip()
            data.append(txt)
    data = np.array(data)
    return data 

def process_data(data, var_cols, var_values):
    IGNORE_VARS = [
        'HHWT',   'SERIAL', 'CBSERIAL', 'PERNUM', 'PERWT', 'IND',  'INDNAICS', 'FTOTINC', 'INCWAGE', 'INCSS', 'PWCOUNTY',
        'SAMPLE', 'GQ',  'INCRETIR', "YEAR", "YRMARR", "YRNATUR", "YRSUSA1", "YRSUSA2", "HINSIHS", "EMPSTAT", "RACE"
    ]
    AS_IS_VARS = [
        'BIRTHYR', # Year in which respondent was born
        'YRMARR', # Year in which respondent was married
        'INCTOT', # each respondent's total pre-tax personal income or losses from all sources for the previous year  
        'INCWELFR', # income from welfare programs
        'INCINVST', # how much pre-tax money the respondent received or lost during the previous year in the form of income from an estate or trust
        'POVERTY', # 3-digit numeric code expressing each family's total income for the previous year as a percentage of the poverty thresholds 
        'YRSUSA1', # 2-digit numeric code reporting how long a person who was born in a foreign country or U.S. outlying area had been living in the United States
        'YEAR',
        'YRIMMIG'  # year of immigration
    ]

    data_dict = defaultdict(list)
    for record in tqdm(data):
        for var, indices in var_cols.items():
            if var in IGNORE_VARS:
                continue
            code = record[indices[0]-1:indices[-1]]
            if var in AS_IS_VARS:
                value = code
                if value == '0000' and var in ['YRMARR']:
                    value = 'N/A'
            else:
                try:
                    value = var_values[var][code]
                except:
                    print(var, code)
            data_dict[var].append(value)
    
    df = pd.DataFrame.from_dict(data_dict)
    return df 


def create_labor(df):
    print("extracting labor task")
    filtered_df = df[df['HINSEMP'] != 'N/A']
    filtered_df = filtered_df.rename(columns = {'HINSEMP': 'HEALTH_LABEL'})
    filtered_df['HEALTH_LABEL'] = filtered_df['HEALTH_LABEL'] == 'Has insurance through employer/union'
    return filtered_df

def create_income(df):
    print("extracting income task")
    # Task 2 - predict income 
    incomes_text = df['INCTOT']
    incomes = np.array([float(x) for x in incomes_text])
    
    df['INCTOT'] = incomes >= 30000
    df = df.rename(columns = {'INCTOT': 'INCOME_LABEL'})
    return df
        
def filter_features(df):
    print("Discretizing Age")
    # discretize age 
    age_strs = df['AGE'].values
    is_digit = [age.isdigit() for age in age_strs]
    df = df[is_digit]
    ages = df['AGE'].values
    df.loc[:, 'AGE'] = [int(int(x) / 10) for x in ages]

    cols = [
        'HISPAN', 'HISPAND', 'BPLD', 'HCOVANY', 'HCOVPRIV', 'HCOVPUB', 
        'HINSCAID', 'HINSCARE', 'HINSVA', 'EDUCD', 'GRADEATTD', 'DEGFIELDD', 
        'DEGFIELD2', 'DEGFIELD2D', 'EMPSTATD', 'LABFORCE', 'MIGCOUNTY1',
        'VETSTATD' , 'VET01LTR', 'VET01LTR', 'VET90X01', 'VET75X90', 'VETVIETN',
        'VET55X64', 'VETKOREA', 'VET47X50', 'VETWWII'
    ]

    cols = [c for c in cols if c in df.columns]
    print("Removing:", cols)
    df = df.drop(columns=cols)

    prefixes = {
        'MARRNO': 'MARRNO',
        'MARST': 'MARST',
        'RACED': 'race',
        'AGE': 'age',
        'SEX': 'gender',
        'BPL': 'BPL',
        'CITIZEN': 'CITIZEN',
        'SCHOOL': 'SCHOOL',
        'EDUC': 'EDUC',
        'CLASSWKR': 'CLASSWKR',
        'CLASSWKRD': 'CLASSWKRD',
        'WKSWORK2': 'WKSWORK2',
        'MIGPLAC1': 'MIGPLAC1',
        'VETDISAB': 'VETDISAB',
        'DIFFREM': 'DIFFREM',
        'DIFFREM': 'DIFFREM',
        'DIFFPHYS': 'DIFFPHYS',
        'DIFFMOB': 'DIFFMOB',
        'DIFFCARE': 'DIFFCARE',
        'DIFFSENS': 'DIFFSENS',
        'DIFFEYE': 'DIFFEYE',
        'DIFFHEAR': 'DIFFHEAR',
        'VETSTAT': 'VETSTAT',
        'PWSTATE2': 'PWSTATE2',
        'TRANWORK': 'TRANWORK',
        'GRADEATT': 'GRADEATT',
        'OCC': 'OCC',
        'SCHLTYPE': 'SCHLTYPE',
        'DEGFIELD': 'DEGFIELD'
    }

    df = pd.get_dummies(df,prefix=prefixes, columns = [p for p in prefixes.keys()])
    return df 
    

def split(df):
    shuffled = df.sample(frac = 1, random_state=0)
    train = shuffled.head(int(len(df)*(90/100)))
    test = shuffled.tail(int(len(df)*(10/100)))
    return train, test 

def main():
    data_dir = "../../../data/ipums/"
    var_cols, var_descriptions, var_values = load_codebook(data_dir)
    raw_data = load_raw(data_dir)
    df = process_data(raw_data, var_cols, var_values)
    print("Num samples: %d. Num features: %d" % df.shape)

    # Create labor task
    df = create_labor(df)

    # Create income task 
    df = create_income(df)

    # filter features 
    df = filter_features(df)

    # split into train/test 
    train, test = split(df)

    # save to file 
    train.to_pickle(os.path.join(data_dir, "train"))
    test.to_pickle(os.path.join(data_dir, "test"))








if __name__ == "__main__":
    main()
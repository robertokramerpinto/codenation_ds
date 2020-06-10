import pandas as pd
import numpy as np
import os
#∫from loan_prediction.loads.loading_data import load_data
from loan_prediction.utils.logger import get_logger
from loan_prediction.utils.cfg import load_config
import sys


# -------------------------------------------------------------------------------------------------
    # 1. GLOBAL VARIABLES
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
    # 2. AUXILIARY FUNCTIONS
# -------------------------------------------------------------------------------------------------

def load_raw_data(cfg, type_dataset):
    if type_dataset == 'train':
        data_path = cfg.general.path.raw_train_file
    else:
        data_path = cfg.general.path.raw_test_file

    df = pd.read_csv(data_path)
    assert len(df) > 0, "Empty DataFrame. Please check data path."
    return df

def save_data(data, cfg, type_dataset):
    if type_dataset == 'train':
        data_path = cfg.general.path.processed_train_file
    else:
        data_path = cfg.general.path.processed_test_file

    data.to_csv(data_path, index=False)
    return None

def preprocess_raw_data(data):

    X = data.copy()

    # Preprocessing Data

    # --- var: Loan Status ---
    var = 'Loan_Status'
    map_dict = {'Y':1, 'N':0}
    X[var] = X[var].map(map_dict)

    # --- var: Dependents ---
    var = 'Dependents'
    map_dict = {'0':0, '1':1, '2':2, '3+':3}
    X[var] = X[var].map(map_dict)

    # --- Education ---
    var = 'Education'
    map_dict = {'Graduate': 1, 'Not Graduate':0}
    X[var] = X[var].map(map_dict)

    # --- CoapplicantIncome ---
    var = 'ApplicantIncome'
    X[var] = np.where(X[var] < 0, 0, X[var])

    # --- CoapplicantIncome ---
    var = 'CoapplicantIncome'
    X[var] = np.where(X[var] < 0, 0, X[var])

    # --- LoanAmount ---
    var = 'LoanAmount'
    X[var] = X[var]*1000


    # Feature Creation

    #a. Total Amount Income
    new_var = 'total_income'
    X[new_var] = X['ApplicantIncome'] + X['CoapplicantIncome']

    # a.1 Total Amount Income
    new_var = 'total_annual_income'
    X[new_var] = X['total_income'] * 12

    #b. %contribution applicant
    new_var = '%_applicant'
    X[new_var] = X['ApplicantIncome']/(X['total_income']+0.1)

    #c. Loan_term
    new_var = 'term'
    def get_term(x):
        if pd.isnull(x):
            return x
        if x < 180:
            return 120
        elif x == 180.0:
            return 180
        elif x <= 300:
            return 300
        elif x <= 360:
            return 360
        else:
            return 480

    X[new_var] = X['Loan_Amount_Term'].apply(get_term)

    #d. single monther
    new_var = 'FL_single_mother'
    def is_single_mother(gender,married,dependents):
        try:
            if gender == 'Female' and married == 'No' and dependents > 0:
                return 1
            else:
                return 0
        except:
            return 0
    X[new_var] = np.vectorize(is_single_mother)(X['Gender'], X['Married'], X['Dependents'])

    #e. income per person
    new_var = 'income_per_capita'
    def get_income_per_capita(married, dependents, annual_income):
        try:
            if married == 'Yes':
                m = 2
            else:
                m = 1
            total_members = m + dependents
            x = annual_income/total_members
            return x
        except:
            return None

    X[new_var] = np.vectorize(get_income_per_capita)(X['Married'], X['Dependents'], X['total_annual_income'])

    # f. %_amount
    new_var = '%_amount'
    def get_annual_amount_perc(total_income, terms, loan_amount):
        try:
            term_years = terms/12
            annual_amount = loan_amount/term_years
            perc_loan = annual_amount / total_income
            return perc_loan
        except:
            return None

    X[new_var] = np.vectorize(get_annual_amount_perc)(X['total_annual_income'], X['Loan_Amount_Term'], X['LoanAmount'], )

    return X


# -------------------------------------------------------------------------------------------------
    # 3. MAIN FUNCTION
# -------------------------------------------------------------------------------------------------

def run():
    # Getting Dataset type argument
    try:
        type_dataset = str(sys.argv[1]).lower()
    except:
        raise ValueError("This script requires an argument when executed ('train' or 'test').")
    if type_dataset not in ['train', 'test']:
        raise ValueError("Please enter one of the following accepted arguments: train, test")

    # Get Configuration
    cfg = load_config()

    # Get Logger
    log = get_logger(cfg.general.path.logs, level=cfg.general.logging.log_level,
                     file_name=str(os.path.basename(__file__)))

    try:
        # Load Data
        log.info("." * 30)
        log.info("Loading Raw Data")
        data = load_raw_data(cfg, type_dataset)

        # Pre-process Raw Data
        log.info("." * 30)
        log.info("Processing Raw Data")
        processed_df = preprocess_raw_data(data)

        # Persist Data
        log.info("." * 30)
        log.info("Persisting Processed Data")
        save_data(processed_df,cfg, type_dataset)

        log.info("." * 30)
        log.info("Script Sucessfully Executed !!")
    except Exception as e:
        log.exception(e)
        raise

# -------------------------------------------------------------------------------------------------
    # 4. EXECUTION
# -------------------------------------------------------------------------------------------------
if __name__=='__main__':
    run()

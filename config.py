"""
Configuration file for the project. Contains file paths, parameters, and other constants.
"""

import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
DATA_NAME = "home-credit-credit-risk-model-stability"
DATA_TRAIN = DATA_DIR / DATA_NAME / "parquet_files" / "train"
DATA_TEST = DATA_DIR / DATA_NAME / "parquet_files" / "test"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOG_DIR = ROOT_DIR / "logs"
SIMILAR_VARIABLES_ANALYSIS = ROOT_DIR / "similar_variables_analysis.csv"

# Create directories if they don't exist
for directory in [DATA_DIR, DATA_TRAIN, DATA_TEST, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TRAIN_BASE = DATA_TRAIN / "train_base.parquet"
TEST_BASE = DATA_TEST / "test_base.parquet"
FEATURE_DEFINITIONS = DATA_DIR / DATA_NAME / "feature_definitions.csv"

# Table groups by depth of train data
DEPTH_0_TABLES_TRAIN = ["static_0_0", "static_0_1", "static_cb_0"]

DEPTH_1_TABLES_TRAIN = [
    "applprev_1_0", "applprev_1_1", "other_1", "tax_registry_a_1", 
    "tax_registry_b_1", "tax_registry_c_1", "credit_bureau_a_1_0", 
    "credit_bureau_a_1_3", "credit_bureau_a_1_1", "credit_bureau_a_1_2", 
    "credit_bureau_b_1", "credit_bureau_b_1", "deposit_1", "person_1", "debitcard_1"
]

DEPTH_2_TABLES_TRAIN = [
    "applprev_2", "person_2", "credit_bureau_a_2_0", "credit_bureau_a_2_1", 
    "credit_bureau_a_2_2", "credit_bureau_a_2_3", "credit_bureau_a_2_4", 
    "credit_bureau_a_2_5","credit_bureau_b_2"
]
# "credit_bureau_a_2_6", "credit_bureau_a_2_7", 
#     "credit_bureau_a_2_8", "credit_bureau_a_2_9", "credit_bureau_a_2_10"

# Table groups by depth of test data
DEPTH_0_TABLES_TEST = ["static_0_0", "static_0_1", "static_0_2", "static_cb_0"]

DEPTH_1_TABLES_TEST = [
    "applprev_1_0", "applprev_1_1", "applprev_1_2", "other_1", "tax_registry_a_1", 
    "tax_registry_b_1", "tax_registry_c_1", "credit_bureau_a_1_0", 
    "credit_bureau_a_1_1", "credit_bureau_a_1_2", "credit_bureau_a_1_3",
    "credit_bureau_a_1_4", "credit_bureau_b_1", "deposit_1", "person_1", "debitcard_1"
]

DEPTH_2_TABLES_TEST = [
    "applprev_2", "person_2", "credit_bureau_a_2_0", "credit_bureau_a_2_1", 
    "credit_bureau_a_2_2", "credit_bureau_a_2_3", "credit_bureau_a_2_4", 
    "credit_bureau_a_2_5",  "credit_bureau_b_2"
]

# "credit_bureau_a_2_6", "credit_bureau_a_2_7", 
#     "credit_bureau_a_2_8", "credit_bureau_a_2_9", "credit_bureau_a_2_10", 
#     "credit_bureau_a_2_11",

# Feature transformation types
TRANSFORMATION_TYPES = {
    'P': 'Transform DPD (Days past due)',
    'M': 'Masking categories',
    'A': 'Transform amount',
    'D': 'Transform date',
    'T': 'Unspecified Transform',
    'L': 'Unspecified Transform'
}

# EDA parameters
MISSING_THRESHOLD = 0.85  # Features with missing values above this threshold may be dropped
CORRELATION_THRESHOLD = 0.7  # Threshold for high correlation between features
CARDINALITY_THRESHOLD = 100  # Threshold for high cardinality categorical features

STABILITY_WEIGHT = 0.5  # Weight for stability in feature selection (vs. performance)

# Chunk size for reading data
CHUNK_SIZE = 1000000

# Feature uses for aggregation

# STATIC_0_USES = [
#     "case_id","avgdbddpdlast24m_3658932P", "avgdbddpdlast3m_4187120P", "maxdpdlast12m_727P",
#     "numinstpaidearly3d_3546850L", "numinstpaidlate1d_3546852L", "numinstls_657L",
#     "currdebt_22A", "credamount_770A", "maxoutstandbalancel12m_4187113A",
#     "avgoutstandbalancel6m_4187114A", "numinstpaid_4499208L", "numrejects9m_859L",
#     "applicationcnt_361L", "maininc_215A", "totaldebt_9A", "maxdpdlast24m_143P",
#     "maxdpdlast6m_474P", "numpmtchanneldd_318L", "applications30d_658L"
# ]

# CREDIT_BUREAU_A_1_USES = [
#     "case_id",'classificationofcontr_13M', 'classificationofcontr_400M', 'contractst_545M', 'contractst_964M',
#     'description_351M', 'financialinstitution_382M', 'financialinstitution_591M', 'purposeofcred_426M',
#     'purposeofcred_874M', 'subjectrole_182M', 'subjectrole_93M', 'refreshdate_3813885D', 'dpdmax_757P',
#     'outstandingamount_354A', 'overdueamount_31A', 'overdueamountmax_35A', 'numberofoverdueinstls_834L',
#     'totalamount_6A', 'dateofcredend_353D', 'dateofcredstart_181D', "num_group1"
# ]

# CREDIT_BUREAU_B_1_USES = [
#     "case_id", "classificationofcontr_1114M", "contractst_516M", "contracttype_653M", "credor_3940957M", "pmtmethod_731M", 
#     "purposeofcred_722M", "subjectrole_326M", "subjectrole_43M", "num_group1", "amount_1115A", "debtvalue_227A", 
#     "credlmt_1052A", "residualamount_127A", "interestrateyearly_538L", "credlmt_3940954A", "residualamount_3940956A",   
#     "credquantity_1099L", "dpd_550P", "installmentamount_833A", "totalamount_503A"
#     # # Feature Interactions
#     # "credlmt_to_totalamount_ratio",  # credlmt_1052A / totalamount_503A
#     # "debt_to_credit_ratio",          # debtvalue_227A / amount_1115A
#     # "interest_to_installment_ratio", # interestrateyearly_538L / installmentamount_833A
# ]

# CREDIT_BUREAU_A_2_USES = [
#     "case_id",
#     "pmts_dpd_303P",  # Days past due of the payment for terminated contract
#     "pmts_overdue_1152A",  # Overdue payment for a closed contract
#     "pmts_month_158T",  # Month of payment for a closed contract
#     "pmts_year_1139T",  # Year of payment for an active contract
#     "pmts_month_706T"  # Month of payment for active contract
#     "num_group1",
#     "num_group2"
# ]

EXCLUDES=["case_id","num_group1","num_group2"]

FRACTION_OF_DATA = 0.2  # Fraction of data to use for training and testing

VIF_THRESHOLD = 10  # Threshold for VIF in multicollinearity check

MULTICOLLINEARITY_THRESHOLD = 0.8  # Threshold for high correlation between pairs of features

TOP_K_FEATS = 10  # Number of top features to display in EDA

KEY_COLUMNS = ["case_id", "WEEK_NUM", "target"]  # Key columns in the data

SAMPLE_SIZE = 300000 

DOMAIN_KNOWLEDGE = False
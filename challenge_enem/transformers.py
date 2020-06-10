import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.sparse.csr import csr_matrix

# Importing Sklearn Base Classes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer, RobustScaler, \
    MinMaxScaler

import sys
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
import multiprocessing
import pandas as pd
import math
import platform

# from sklearn.preprocessing import PowerTransformer,KBinsDiscretizer

"""
0. New test labels
1. Transform modify create new variables (exclusive code+class)
2. missing values (num + cat)
3. num transform + binning 
4. outliers
5. cat encoding
6. scaling
"""


##########################
#### Aux Functions
##########################

def convert_cols_to_list(cols):
    if isinstance(cols, pd.Series):
        return cols.tolist()
    elif isinstance(cols, np.ndarray):
        return cols.tolist()
    elif np.isscalar(cols):
        return [cols]
    elif isinstance(cols, set):
        return list(cols)
    elif isinstance(cols, tuple):
        return list(cols)
    elif pd.api.types.is_categorical(cols):
        return cols.astype(object).tolist()

    return cols


def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


def convert_input(X, columns=None, deep=False):
    """
    Unite data into a DataFrame.
    Objects that do not contain column names take the names from the argument.
    Optionally perform deep copy of the data.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, copy=deep)
        else:
            if columns is not None and np.size(X, 1) != len(columns):
                raise ValueError('The count of the column names does not correspond to the count of the columns')
            if isinstance(X, list):
                X = pd.DataFrame(X, columns=columns,
                                 copy=deep)  # lists are always copied, but for consistency, we still pass the argument
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X, columns=columns, copy=deep)
            elif isinstance(X, csr_matrix):
                X = pd.DataFrame(X.todense(), columns=columns, copy=deep)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))
    elif deep:
        X = X.copy(deep=True)

    return X


def convert_input_vector(y, index):
    """
    Unite target data type into a Series.
    If the target is a Series or a DataFrame, we preserve its index.
    But if the target does not contain index attribute, we use the index from the argument.
    """
    if y is None:
        raise ValueError('Supervised encoders need a target for the fitting. The target cannot be None')
    if isinstance(y, pd.Series):
        return y
    elif isinstance(y, np.ndarray):
        if len(np.shape(y)) == 1:  # vector
            return pd.Series(y, name='target', index=index)
        elif len(np.shape(y)) == 2 and np.shape(y)[0] == 1:  # single row in a matrix
            return pd.Series(y[0, :], name='target', index=index)
        elif len(np.shape(y)) == 2 and np.shape(y)[1] == 1:  # single column in a matrix
            return pd.Series(y[:, 0], name='target', index=index)
        else:
            raise ValueError('Unexpected input shape: %s' % (str(np.shape(y))))
    elif np.isscalar(y):
        return pd.Series([y], name='target', index=index)
    elif isinstance(y, list):
        if len(y) == 0 or (len(y) > 0 and not isinstance(y[0], list)):  # empty list or a vector
            return pd.Series(y, name='target', index=index, dtype=float)
        elif len(y) > 0 and isinstance(y[0], list) and len(y[0]) == 1:  # single row in a matrix
            flatten = lambda y: [item for sublist in y for item in sublist]
            return pd.Series(flatten(y), name='target', index=index)
        elif len(y) == 1 and len(y[0]) == 0 and isinstance(y[0], list):  # single empty column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=float)
        elif len(y) == 1 and isinstance(y[0], list):  # single column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=type(y[0][0]))
        else:
            raise ValueError('Unexpected input shape')
    elif isinstance(y, pd.DataFrame):
        if len(list(y)) == 0:  # empty DataFrame
            return pd.Series(name='target', index=index, dtype=float)
        if len(list(y)) == 1:  # a single column
            return y.iloc[:, 0]
        else:
            raise ValueError('Unexpected input shape: %s' % (str(y.shape)))
    else:
        return pd.Series(y, name='target', index=index)  # this covers tuples and other directly convertible types


def get_generated_cols(X_original, X_transformed, to_transform):
    """
    Returns a list of the generated/transformed columns.
    Arguments:
        X_original: df
            the original (input) DataFrame.
        X_transformed: df
            the transformed (current) DataFrame.
        to_transform: [str]
            a list of columns that were transformed (as in the original DataFrame), commonly self.cols.
    Output:
        a list of columns that were transformed (as in the current DataFrame).
    """
    original_cols = list(X_original.columns)

    if len(to_transform) > 0:
        [original_cols.remove(c) for c in to_transform]

    current_cols = list(X_transformed.columns)
    if len(original_cols) > 0:
        [current_cols.remove(c) for c in original_cols]

    return current_cols


class TransformerWithTargetMixin:
    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
             transform(X, y)
        and not with:
            transform(X)
        """
        if y is None:
            raise TypeError('fit_transform() missing argument: ''y''')
        return self.fit(X, y, **fit_params).transform(X, y)


def imput_missing_value(x, var, label, dictionary):
    """Impute missing values based on a mapping dicitionary"""
    if pd.isnull(x) == True:
        return dictionary[var][label]
    else:
        return x


####################################################
#### 1. Missing Values Imputation
####################################################

# 1.1 NUMERICAL VARIABLES

# 1.1.1 MEDIAN IMPUTATION

class NumericalMedianImputer(BaseEstimator, TransformerMixin):
    """ Numerical Missing Value Imputation. Replaces Null values by the median value.
        Also allows the addition of a binary column to identify missing values.
        The Median can be obtained considering the entire dataset or grouping info by a given feature.

    Parameters
    ----------
    variables (list): List of numerical variables to be imputed
    groupby_var (str): categorical feature to be used to segment data and extract the medians for each category
    add_column (bool): boolean to add extra binary column to identify missing values
    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables, groupby_var=None, add_column=True):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

        self.groupby_var = groupby_var
        self.add_column = add_column

    def fit(self, X, y=None):
        self.numerical_imputer_dict = {}

        if self.groupby_var:  # groupby_var is defined
            self.unique_categories_groupby_var = X[self.groupby_var].unique().tolist()
            for var in self.variables:
                self.numerical_imputer_dict[var] = {}

                tmp = X.groupby(self.groupby_var)[var].median()
                for category in self.unique_categories_groupby_var:
                    freq_value = tmp.loc[category]
                    self.numerical_imputer_dict[var][category] = freq_value
        else:
            for var in self.variables:
                self.numerical_imputer_dict[var] = X[var].median()

        return self

    def transform(self, X):
        X = X.copy()

        if self.groupby_var:
            for var in self.variables:
                if self.add_column:
                    X[var + '_FL_MISSING'] = np.where(pd.isnull(X[var]) == True, 1, 0)

                X[var] = np.vectorize(imput_missing_value)(X[var],
                                                           var,
                                                           X[self.groupby_var],
                                                           self.numerical_imputer_dict)
        else:
            for var in self.variables:
                if self.add_column:
                    X[var + '_FL_MISSING'] = np.where(pd.isnull(X[var]) == True, 1, 0)

                X[var] = X[var].fillna(self.numerical_imputer_dict[var])

        return X


# 1.1.1 ARBITRARY VALUES IMPUTATION

class NumericalNAImputerValue(BaseEstimator, TransformerMixin):
    """ Numerical Missing Value Transformation.
        Replaces Null values by an arbitrary given value.

    Parameters
    ----------
    variables (list): List of numerical variables to be imputed
    value (int): Imput value for replacement (0 by default)
    add_column(bool): True by default. Adds a binary column to indicate the presence of a missing value

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables, value=0, add_column=True):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)
        self.value = value
        self.add_column = add_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if self.add_column:
                X[var + '_FL_MISSING'] = np.where(pd.isnull(X[var]) == True, 1, 0)

            X[var] = X[var].fillna(float(self.value))

        return X


# 1.2 CATEGORICAL VARIABLES

# 1.2.1 MODE (MOST FREQUENT LABEL) IMPUTATION

class CategoricalMostFrequentImputer(BaseEstimator, TransformerMixin):
    """ Categorical Missing Value Imputation.
        Replaces Null values by the most frequent observed label.
        Can also add a binary column to detect missing values.
        The most frequent label can be obtained considering the entire dataset or grouping info by a given feature.

    Parameters
    ----------
    variables (list): List of categorical variables to be analyzed
    groupby_var (str): catgorical feature to be used to segment data and extrac the most frequent label for each category
    add_column (bool): boolean to add extra binary column to identify missing values

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables, groupby_var=None, add_column=True):
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

        self.groupby_var = groupby_var
        self.add_column = add_column

    def fit(self, X, y=None):
        self.categorical_frequent_dict = {}  # dict to store frequent labels for each variable

        if self.groupby_var:  # groupby_var is defined
            self.unique_categories_groupby_var = X[self.groupby_var].unique().tolist()
            for var in self.variables:
                self.categorical_frequent_dict[var] = {}

                tmp = X.groupby(self.groupby_var)[var].value_counts()
                for category in self.unique_categories_groupby_var:
                    freq_value = tmp.loc[category].index[0]
                    self.categorical_frequent_dict[var][category] = freq_value
        else:
            for var in self.variables:
                freq_value = X[var].value_counts().index[0]
                self.categorical_frequent_dict[var] = freq_value

        return self

    def transform(self, X):
        X = X.copy()

        if self.groupby_var:
            for var in self.variables:
                if self.add_column:
                    X[var + '_FL_MISSING'] = np.where(pd.isnull(X[var]) == True, 1, 0)
                X[var] = np.vectorize(imput_missing_value)(X[var],
                                                           var,
                                                           X[self.groupby_var],
                                                           self.categorical_frequent_dict)
        else:
            for var in self.variables:
                if self.add_column:
                    X[var + '_FL_MISSING'] = np.where(pd.isnull(X[var]) == True, 1, 0)
                X[var] = X[var].fillna(self.categorical_frequent_dict[var])

        return X


class CategoricalImputerLabel(BaseEstimator, TransformerMixin):
    """ Categorical Missing Value Transformation.
        Replaces Null values by a given label (default = 'MISSING')

    Parameters
    ----------
    variables (list): List of categorical variables to be analyzed
    replace_str (str): Label to replace missing values ('MISSING' is the default value)

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables=None, replace_str='MISSING'):
        if not isinstance(variables, list):  # checking variables type
            self.variables = list(variables)
        else:
            self.variables = variables
        self.replace_str = replace_str

    # Fit
    def fit(self, X, y=None):
        return self

    # Transform
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.replace_str)

        return X


####################################################
#### 2. CATEGORICAL ENCODING
####################################################

# 2.1 OHE - ONE HOT ENCODING (K variables)

class CategoricalEncoderOHE(BaseEstimator, TransformerMixin):
    """ Apply One Hot Encoding transformation.
        Categorical variables are transformed into k dummy variables
        where k is the distinct number of labels.

    Parameters
    ----------
    prefix_separator (str): prefix separator used to create the dummy variables ('_CAT_' default)

    Returns
    -------
    pd.DataFrame: Transformed DataFrame Object


    """

    def __init__(self, prefix_separator='_CAT_', replace_new=False):
        self.prefix_separator = prefix_separator
        self.replace_new = replace_new
        self.variable_labels = {}
        self.final_columns = []
        self.cat_variables = []
        self.variable_mode = {}

    def fit(self, X, y=None):

        if self.replace_new:
            # get all categorical variables
            self.cat_variables = [col for col in X.columns if 'OBJECT' in str(X[col].dtype).upper()]
            # get all labels from the each categorical variables
            for var in self.cat_variables:
                self.variable_labels[var] = X[var].unique().tolist()
                # get the most frequent value
                self.variable_mode[var] = X[var].mode()[0]

        # Saving labels in the training set
        tmp = pd.get_dummies(X, prefix_sep=self.prefix_separator)
        self.final_columns = tmp.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()

        if self.replace_new:
            for var in self.cat_variables:
                labels = self.variable_labels[var]
                mode = self.variable_mode[var]
                control = np.where(X[var].isin(labels) == False, 1, 0)
                if np.sum(control) > 0:
                    print(
                        "Variable '{}' presents new labels in test set. \nAll new labels have been replaced by the most frequent label from train set: '{}'".format(
                            var, mode))
                    print(X[X[var].isin(labels) == False][var].unique())
                    print("-" * 15)
                X[var] = np.where(X[var].isin(labels) == False, mode, X[var])

        # Adjusting for labels that are not present in the Test Set
        X = pd.get_dummies(X, prefix_sep=self.prefix_separator)
        for col in self.final_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.final_columns]
        return X


# 2.2 CATEGORICAL LABEL ENCODER

class CategoricalEncoderLabel(BaseEstimator, TransformerMixin):
    """ Apply Categorical Label Encoder from Sklearn.
        Option to replace new labels by most frequent ones.

    Parameters
    ----------
    variables (list): List with categorical variables
    replace_new (boolean): Replaces new labels in test by most frequent (train)

    Returns
    -------
    pd.DataFrame: Transformed DataFrame Object (categorical variables are returned as category type)


    """

    def __init__(self, variables=None, replace_new=False):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.replace_new = replace_new
        self.variable_labels = {}
        self.final_columns = []
        self.cat_variables = []
        self.variable_mode = {}
        self.label_encoder_dict = {}

    def fit(self, X, y=None):

        if self.replace_new:
            # get all categorical variables
            self.cat_variables = [col for col in X.columns if 'OBJECT' in str(X[col].dtype).upper()]
            # get all labels from the each categorical variables
            for var in self.cat_variables:
                self.variable_labels[var] = X[var].unique().tolist()
                # get the most frequent value
                self.variable_mode[var] = X[var].mode()[0]

        for var in self.variables:
            X[var] = X[var].apply(lambda x: str(x))
            self.label_encoder_dict[var] = {}
            encoder = LabelEncoder().fit(X[var])
            for category in encoder.classes_:
                self.label_encoder_dict[var] = encoder

        return self

    def transform(self, X):
        X = X.copy()
        if self.replace_new:
            for var in self.cat_variables:
                labels = self.variable_labels[var]
                mode = self.variable_mode[var]
                control = np.where(X[var].isin(labels) == False, 1, 0)
                if np.sum(control) > 0:
                    print(
                        "Variable '{}' presents new labels in test set. \nAll new labels have been replaced by the most frequent label from train set: '{}'".format(
                            var, mode))
                    print(X[X[var].isin(labels) == False][var].unique())
                    print("-" * 15)
                X[var] = np.where(X[var].isin(labels) == False, mode, X[var])

        for var in self.variables:
            encoder = self.label_encoder_dict[var]
            X[var] = encoder.transform(X[var])
            X[var] = X[var].astype('category')
        return X


# 2.3 TARGET ENCODING

class CategoricalTargetEncoder(TransformerMixin):
    """ Target encoder for Categorical variables.
    Treats missing values by creating a new label for them ('MISSING').
    Missing values or new categories in test set are assigned with global mean value.
    Parameters
    ----------
    variables (list): list of categorical variables to be encoded
    min_sample (int): minimum samples to consider the label.
    min_sample_count_pos_only (boolean): consider pos prior effect
    smoothing_param (float): smoothing effect to balance average vs prior means
    Returns
    -------
    pd.DataFrame: Transformed DataFrame Object
    """

    def __init__(self, variables, min_samples=1, min_sample_count_pos_only=False, smoothing_param=200):
        self.columns = variables
        self.min_samples = min_samples
        self.smoothing_param = smoothing_param

        if min_sample_count_pos_only:
            self.aggfun = "sum"
        else:
            self.aggfun = "count"

    def transform(self, X, *_):
        X = X.copy()
        X[self.columns] = X[self.columns].fillna("MISSING")

        for col in self.columns:
            X[col] = X[col].map(self.encoding[col]).fillna(self.prior)
        return X

    def fit(self, X, y):

        self.prior = y.mean()
        encoding = {}

        df = X.loc[:, self.columns]
        df = df.fillna("MISSING")
        df["y"] = y

        for col in self.columns:
            replacement_stat = df.groupby([col])["y"].agg(["mean", self.aggfun])
            smoothing = 1 / (1 + np.exp(- (replacement_stat['count'] - self.min_samples) / self.smoothing_param))
            enc = self.prior * (1 - smoothing) + replacement_stat['mean'] * smoothing
            encoding[col] = enc.to_dict()

        self.encoding = encoding
        return self

    def get_params(self, deep=True):

        return {"columns": self.columns, "min_samples": self.min_samples, "smoothing_param": self.smoothing_param}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # 2.4 ORDINALCATEGORICALENCODER


class OrderedCategoricalEncoder(TransformerMixin):
    """ Ordered Categorical Encoder for categorical variables.
    Creates numerical integer labels for each category from 0 to k-1
    where k is the number of distinct categories for a given variable.
    The order is obtained accordingly to the target mean.
    0 --> higher mean
    k-1 --> lowest mean

    Parameters
    ----------
    variables (list): List with categorical variables

    Returns
    -------
    pd.DataFrame: Transformed DataFrame Object (categorical variables are returned as category type)


    """

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.encoding_dict = {}

    def fit(self, X, y=None):
        tmp = X.copy()
        tmp['y'] = y

        for col in self.variables:
            tmp_groupby = tmp.groupby(col)['y'].mean().sort_values(ascending=False).reset_index().reset_index()
            self.encoding_dict[col] = dict(zip(tmp_groupby[col], tmp_groupby['index']))
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.variables:
            X[col] = X[col].map(self.encoding_dict[col])

        return X

    """The hashing module contains all methods and classes related to the hashing trick."""


class HashingEncoder(BaseEstimator, TransformerMixin):
    """ A multivariate hashing implementation with configurable dimensionality/precision.
    The advantage of this encoder is that it does not maintain a dictionary of observed categories.
    Consequently, the encoder does not grow in size and accepts new values during data scoring
    by design.
    It's important to read about how max_process & max_sample work
    before setting them manually, inappropriate setting slows down encoding.
    Default value of 'max_process' is 1 on Windows because multiprocessing might cause issues, see in :
    https://github.com/scikit-learn-contrib/categorical-encoding/issues/215
    https://docs.python.org/2/library/multiprocessing.html?highlight=process#windows
    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    hash_method: str
        which hashing method to use. Any method from hashlib works.
    max_process: int
        how many processes to use in transform(). Limited in range(1, 64).
        By default, it uses half of the logical CPUs.
        For example, 4C4T makes max_process=2, 4C8T makes max_process=4.
        Set it larger if you have a strong CPU.
        It is not recommended to set it larger than is the count of the
        logical CPUs as it will actually slow down the encoding.
    max_sample: int
        how many samples to encode by each process at a time.
        This setting is useful on low memory machines.
        By default, max_sample=(all samples num)/(max_process).
        For example, 4C8T CPU with 100,000 samples makes max_sample=25,000,
        6C12T CPU with 100,000 samples makes max_sample=16,666.
        It is not recommended to set it larger than the default value.
    n_components: int
        how many bits to use to represent the feature. By default we use 8 bits.
        For high-cardinality features, consider using up-to 32 bits.
    """

    def __init__(self, max_process=0, max_sample=0, verbose=0, n_components=8, cols=None, drop_invariant=False,
                 return_df=True, hash_method='md5'):

        if max_process not in range(1, 128):
            if platform.system == 'Windows':
                max_process = 1
            else:
                self.max_process = int(math.ceil(multiprocessing.cpu_count() / 2))
                if self.max_process < 1:
                    self.max_process = 1
                elif self.max_process > 128:
                    self.max_process = 128
        else:
            self.max_process = max_process
        self.max_sample = int(max_sample)
        self.auto_sample = max_sample <= 0
        self.data_lines = 0
        self.X = None

        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols
        self.hash_method = hash_method
        self._dim = None
        self.feature_names = None

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)
        else:
            self.cols = convert_cols_to_list(self.cols)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                          "Not found in generated cols.\n{}".format(e))

        return self

    @staticmethod
    def require_data(self, data_lock, new_start, done_index, hashing_parts, cols, process_index):
        if data_lock.acquire():
            if new_start.value:
                end_index = 0
                new_start.value = False
            else:
                end_index = done_index.value

            if all([self.data_lines > 0, end_index < self.data_lines]):
                start_index = end_index
                if (self.data_lines - end_index) <= self.max_sample:
                    end_index = self.data_lines
                else:
                    end_index += self.max_sample
                done_index.value = end_index
                data_lock.release()

                data_part = self.X.iloc[start_index: end_index]
                # Always get df and check it after merge all data parts
                data_part = self.hashing_trick(X_in=data_part, hashing_method=self.hash_method, N=self.n_components,
                                               cols=self.cols)
                if self.drop_invariant:
                    for col in self.drop_cols:
                        data_part.drop(col, 1, inplace=True)
                part_index = int(math.ceil(end_index / self.max_sample))
                hashing_parts.put({part_index: data_part})
                if self.verbose == 5:
                    print("Process - " + str(process_index),
                          "done hashing data : " + str(start_index) + "~" + str(end_index))
                if end_index < self.data_lines:
                    self.require_data(self, data_lock, new_start, done_index, hashing_parts, cols=cols,
                                      process_index=process_index)
            else:
                data_lock.release()
        else:
            data_lock.release()

    def transform(self, X, override_return_df=False):
        """
        Call _transform() if you want to use single CPU with all samples
        """
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        self.X = convert_input(X)
        self.data_lines = len(self.X)

        # then make sure that it is the right size
        if self.X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (self.X.shape[1], self._dim,))

        if not list(self.cols):
            return self.X

        data_lock = multiprocessing.Manager().Lock()
        new_start = multiprocessing.Manager().Value('d', True)
        done_index = multiprocessing.Manager().Value('d', int(0))
        hashing_parts = multiprocessing.Manager().Queue()

        if self.auto_sample:
            self.max_sample = int(self.data_lines / self.max_process)
        if self.max_process == 1:
            self.require_data(self, data_lock, new_start, done_index, hashing_parts, cols=self.cols, process_index=1)
        else:
            n_process = []
            for thread_index in range(self.max_process):
                process = multiprocessing.Process(target=self.require_data,
                                                  args=(
                                                  self, data_lock, new_start, done_index, hashing_parts, self.cols,
                                                  thread_index + 1))
                process.daemon = True
                n_process.append(process)
            for process in n_process:
                process.start()
            for process in n_process:
                process.join()
        data = self.X
        if self.max_sample == 0 or self.max_sample == self.data_lines:
            if hashing_parts:
                data = list(hashing_parts.get().values())[0]
        else:
            list_data = {}
            while not hashing_parts.empty():
                list_data.update(hashing_parts.get())
            sort_data = []
            for part_index in sorted(list_data):
                sort_data.append(list_data[part_index])
            if sort_data:
                data = pd.concat(sort_data, ignore_index=True)
        # Check if is_return_df
        if self.return_df or override_return_df:
            return data
        else:
            return data.values

    def _transform(self, X, override_return_df=False):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X

        X = self.hashing_trick(X, hashing_method=self.hash_method, N=self.n_components, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    @staticmethod
    def hashing_trick(X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
        """A basic hashing implementation with configurable dimensionality/precision
        Performs the hashing trick on a pandas dataframe, `X`, using the hashing method from hashlib
        identified by `hashing_method`.  The number of output dimensions (`N`), and columns to hash (`cols`) are
        also configurable.
        Parameters
        ----------
        X_in: pandas dataframe
            description text
        hashing_method: string, optional
            description text
        N: int, optional
            description text
        cols: list, optional
            description text
        make_copy: bool, optional
            description text
        Returns
        -------
        out : dataframe
            A hashing encoded dataframe.
        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.
        .. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg (2009). Feature Hashing
        for Large Scale Multitask Learning. Proc. ICML.
        """

        try:
            if hashing_method not in hashlib.algorithms_available:
                raise ValueError('Hashing Method: %s Not Available. Please use one from: [%s]' % (
                    hashing_method,
                    ', '.join([str(x) for x in hashlib.algorithms_available])
                ))
        except Exception as e:
            try:
                _ = hashlib.new(hashing_method)
            except Exception as e:
                raise ValueError('Hashing Method: %s Not Found.')

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns.values

        def hash_fn(x):
            tmp = [0 for _ in range(N)]
            for val in x.values:
                if val is not None:
                    hasher = hashlib.new(hashing_method)
                    if sys.version_info[0] == 2:
                        hasher.update(str(val))
                    else:
                        hasher.update(bytes(str(val), 'utf-8'))
                    tmp[int(hasher.hexdigest(), 16) % N] += 1
            return pd.Series(tmp, index=new_cols)

        new_cols = ['col_%d' % d for d in range(N)]

        X_cat = X.loc[:, cols]
        X_num = X.loc[:, [x for x in X.columns.values if x not in cols]]

        X_cat = X_cat.apply(hash_fn, axis=1)
        X_cat.columns = new_cols

        X = pd.concat([X_cat, X_num], axis=1)

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.
        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not isinstance(self.feature_names, list):
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names


####################################################
#### 3. NUMERICAL TRANSFORMATIONS
####################################################

# 3.1 Logarithmic Transformation

class LogarithmicTransformation(BaseEstimator, TransformerMixin):
    """ Apply a Logarithm transformation to data.
        Log transforms are useful when applied to skewed distributions
        as they tend to expand data and make data more normal-like.
    Args:
        variables (list): List of numerical variables to be analyzed
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if (X[var] < 0).sum() > 0:
                print("'{}' presents negative values. Log transformation not applied".format(var))
                X[var] = X[var]
            else:
                X[var] = np.log(1 + X[var])
        return X


# 3.2 BoxCox Transformation

class BoxCoxTransformation(BaseEstimator, TransformerMixin):
    """ Apply a BoxCox statistical transformation to data.
        Uses Scipy.stats BoxCox transformation.
        Data needs to be strictly positive.
    Args:
        variables (list): List of numerical variables to be analyzed
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if (X[var] < 0).sum() > 0:
                print("'{}' presents negative values. Log transformation not applied".format(var))
                X[var] = X[var]
            else:
                X[var], _ = stats.boxcox(X[var])
        return X


class PowerTransformations(BaseEstimator, TransformerMixin):
    """Apply a power transform featurewise to make data more Gaussian-like.
    Apply the Yeo-Johnson transformation that supports negative values

    Args:
        variables(list): list of variables to apply the transformation
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)
        self.transformer = {}

    def fit(self, X, y=None):
        for var in self.variables:
            pt = PowerTransformer(method='yeo-johnson')
            pt.fit(X[[var]])
            self.transformer[var] = pt
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            pt = self.transformer[var]
            X[var] = pd.Series(pt.transform(X[[var]])[:, 0]).values
        return X


####################################################
#### 4. NUMERICAL DISCRETIZATION
####################################################

# 4.1 KBINS DISCRETIZER FROM SKLEARN

class DiscretizerKBinsSklearn(BaseEstimator, TransformerMixin):
    """Bins continuous data into intervals.
    Uses sklearn.preprocessing.KBinsDiscretizer as method.
    Ordinal bin identification: integer value of the bin.

    Parameters
    ----------
    variables(list): list of variables to apply such Transformation.
    strategy(str) --> strategy used to define the widths of the bins
                    'uniform' = all bins in each feature have identical width
                    'quantile' = all bins have the same number of values
                    'kmeans' = Values in each bin have the same nearest center of a 1D k-means cluster

    """

    def __init__(self, variables, n_bins, strategy):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)
        self.n_bins = n_bins
        self.strategy = strategy
        self.transformer = {}

    def fit(self, X, y=None):
        for var in self.variables:
            disc = KBinsDiscretizer(n_bins=self.n_bins, strategy=self.strategy, encode='ordinal')
            disc.fit(X[[var]])
            self.transformer[var] = disc
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            disc = self.transformer[var]
            X[var] = pd.Series(disc.transform(X[[var]])[:, 0]).values

        return X


####################################################
#### 5. HANDLING OUTLIERS
####################################################

# 5.1 Top Bottom capping

class OutliersTopBottom(BaseEstimator, TransformerMixin):
    """ Numerical Outliers treatment.
        Bottom and Top Cap.
        Uses a factor multiplied by the IQR (Interquartile Range).
    Args:
        variables (list): List of numerical variables to be analyzed
        factor (float): factor that multiplies the IQR (1.5 by default)
    """

    def __init__(self, variables, factor=1.5):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)
        self.factor = factor

    def fit(self, X, y=None):
        self.iqr_dict = {}
        for var in self.variables:
            self.iqr_dict[var] = {}
            # calculating top/bottom bounds
            iqr = X[var].quantile(0.75) - X[var].quantile(0.25)
            upper_bound = X[var].quantile(0.75) + self.factor * iqr
            self.iqr_dict[var]['upper_bound'] = upper_bound

            lower_bound = X[var].quantile(0.25) - self.factor * iqr
            self.iqr_dict[var]['lower_bound'] = lower_bound

        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            upper_bound = self.iqr_dict[var]['upper_bound']
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])

            lower_bound = self.iqr_dict[var]['lower_bound']
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])

        return X


# 5.2 Quantiles capping

class OutliersQuantiles(BaseEstimator, TransformerMixin):
    """Capping outliers according to specified quantiles.

    Parameters
    ----------
    variables(list): list of variables to apply such Transformation.
    lower_boundary (float): lower quantile boundary to be used (0.03 is default)
    upper_boundary (float): upper quantile boundary to be used (0.97 is default)

    Returns
    -------
    pd.DataFrame: Transformed DataFrame Object (categorical variables are returned as category type)


    """

    def __init__(self, variables, lower_boundary=0.03, upper_boundary=0.97):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.boundaries_dict = {}

    def fit(self, X, y=None):
        for var in self.variables:
            self.boundaries_dict[var] = {}
            # calculating lower/upper boundaries
            lower = X[var].quantile(self.lower_boundary)
            self.boundaries_dict[var]['lower_bound'] = lower

            upper = X[var].quantile(self.upper_boundary)
            self.boundaries_dict[var]['upper_bound'] = upper
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            upper_bound = self.boundaries_dict[var]['upper_bound']
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])

            lower_bound = self.boundaries_dict[var]['lower_bound']
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])

        return X


####################################################
#### 6. FEATURE SCALING
####################################################

# 6.1 Standardisation

class AdjustedScaler(BaseEstimator, TransformerMixin):
    """Adapts Sklearn Scalers

    Parameters
    ----------
    scaler_type(str):
        'standard' : StandardScaler --> default
        'robust': RobustScaler
        'min_max': MinMaxScaler

    """

    def __init__(self, scaler_type='standard'):
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'min_max':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = X.copy()
        self.data = X
        self.columns = X.columns.tolist()

        self.scaler.fit(self.data)

        return self

    def transform(self, X):
        X = X.copy()
        numpy_data = self.scaler.transform(X)
        df_output = pd.DataFrame(numpy_data, columns=self.columns)
        return df_output


# 6.2 Mean NormalizationStandardisation

class FeatureMeanNormalization(BaseEstimator, TransformerMixin):
    """Applies the Mean Normalization Scaling

    Parameters
    ----------
    variables (list): list with features to be performed the transformation

    """

    def __init__(self):
        self.scaler_mean = StandardScaler(with_mean=True, with_std=False)
        self.scaler_minmax = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(0, 100))

    def fit(self, X, y=None):
        X = X.copy()
        self.data = X
        self.columns = X.columns.tolist()

        self.scaler_mean.fit(self.data)
        self.scaler_minmax.fit(self.data)

        return self

    def transform(self, X):
        X = X.copy()
        numpy_data = self.scaler_minmax.transform(self.scaler_mean.transform(self.data))
        df_output = pd.DataFrame(numpy_data, columns=self.columns)
        return df_output


###################
#### 7. NEW LABELS
###################

# 7.1 TREATS NEW LABELS FOR unseen CATEGORICAL VARIABLES

class TreatNewLabels(TransformerMixin):
    """ Treat Unseen new labels for test sets.

    Parameters
    ----------
    variables (list): List of categorical variables to be treated
    exclude (boolean)
        True --> Exclude unseen observations completely
        False --> Replaces new labels by most frequent ones observed in training set

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables, exclude=False):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

        self.exclude = exclude
        self.labels_values = {}
        self.labels_replace = {}

    def fit(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            #print(var)
            labels = X[X[var].isnull() == False][var].unique().tolist()
            self.labels_values[var] = labels
            #print(labels)

            tmp = pd.DataFrame(X[var].value_counts())
            self.labels_replace[var] = tmp.index[0]
            #print(tmp.index[0])
        return self

    def transform(self, X):
        X = X.copy()
        if self.exclude == False:
            X = X.copy()
            for var in self.variables:
                X[var] = np.where(X[var].isin(self.labels_values[var]) == True, X[var],
                                  np.where(pd.isnull(X[var]) == True, X[var], self.labels_replace[var]))

            return X
        else:
            filter_1 = X[var].isin(self.labels_values[var])
            return X[filter_1]


###################
#### 8. Data Types - Categorical Cols
###################

class ConvertStringType(TransformerMixin):
    """ Convert col to string

    Parameters
    ----------
    variables (list): List of categorical cols to be string types

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].astype(str)
        return X

    
###################
#### 8. Specific Transformations
###################
class SelectFeatures(TransformerMixin):
    def __init__(self, features_list):
        if not isinstance(features_list, list):
            self.features_list = features_list
        else:
            self.features_list = list(features_list)

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        X = X[self.features_list].copy()
        return X
    


class New_Features_Creation(TransformerMixin):
    """ Creates new features based on input data

    Parameters
    ----------
    variables (list): List of categorical cols to be string types

    Returns
    -------
    pd.DataFrame: Imputed DataFrame Object
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = list(variables)

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].astype(str)
        return X

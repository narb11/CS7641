import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import seaborn as sns

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.
        :return: Nothing
        """

        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()
        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        print(train_features)
        print(train_classes)
        return train_features, train_classes

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class BankMarketingData(DataLoader):

    def __init__(self, path='../data/bank-additional-full.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path,sep=';')

    def data_name(self):
        return 'BankMarketingData'

    def class_column_name(self):
        return 'default payment next month'

    def _preprocess_data(self):
        self._data.y.replace(('yes', 'no'), (1, 0), inplace=True)
        print(self._data)

        self._data = utils.shuffle(self._data)
        X = self._data.select_dtypes(include=[object])
        to_encode = X.columns
        print(to_encode)

        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)
        print(self._data)

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes


class WhiteWineData(DataLoader):

    def __init__(self, path='../data/winequality-white.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=';')

    def data_name(self):
        return 'WhiteWineData'

    def class_column_name(self):
        return '12'

    def _preprocess_data(self):
        self._data.quality[self._data.quality < 7] = 0
        self._data.quality[self._data.quality >=7] = 1
        print(self._data.quality.value_counts())
        # print(self._data)


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class ContraceptiveData(DataLoader):

    def __init__(self, path='../data/cmc.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',',names = ["wifes_age","wifes_edu","husbands_edu","num_children","wife_religion","wife_working","husband_occupation","sol_index","media_exp","contraceptive"])

    def data_name(self):
        return 'ContraceptiveData'

    def class_column_name(self):
        return 'contraceptive'

    def _preprocess_data(self):
        self._data.contraceptive[self._data.contraceptive ==1] = 0
        self._data.contraceptive[self._data.contraceptive >=2] = 1

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class BanknoteData(DataLoader):

    def __init__(self, path='../data/data_banknote_authentication.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',',names = ["variance","skewness","kurtosis","entropy","authentic"])

    def data_name(self):
        return 'ContraceptiveData'

    def class_column_name(self):
        return 'contraceptive'

    def _preprocess_data(self):
        self._data.contraceptive[self._data.contraceptive ==1] = 0
        self._data.contraceptive[self._data.contraceptive >=2] = 1

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class EpilepsyData(DataLoader):

    def __init__(self, path='../data/epilepsy.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',')

    def data_name(self):
        return 'EpilepsyData'

    def class_column_name(self):
        return 'epilepsy'

    def _preprocess_data(self):
        self._data.drop(self._data.columns[0], axis=1,inplace=True)
        self._data.y[self._data.y ==1] = 1
        self._data.y[self._data.y ==2] = 0
        self._data.y[self._data.y == 3] = 0
        self._data.y[self._data.y == 4] = 0
        self._data.y[self._data.y == 5] = 0


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class ChessData(DataLoader):

    def __init__(self, path='../data/kr-vs-kp.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',', header=None)

    def data_name(self):
        return 'ChessData'

    def class_column_name(self):
        return 'chess'

    def _preprocess_data(self):
        print(self._data.columns)
        self._data.iloc[:, -1].replace(('won', 'nowin'), (1, 0), inplace=True)
        print(self._data)

        X = self._data.select_dtypes(include=[object])
        to_encode = X.columns
        print(to_encode)

        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)
        print(self._data.iloc[:, -1].value_counts())
        print(self._data)


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class NurseryData(DataLoader):

    def __init__(self, path='../data/nursery.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',',header=None)

    def data_name(self):
        return 'NurseData'

    def class_column_name(self):
        return 'nurse'

    def _preprocess_data(self):
        print(self._data.columns)
        self._data.iloc[:, -1].replace(('recommend', 'priority', 'not_recom','very_recom','spec_prior'), (1,1, 0,1,1), inplace=True)
        print(self._data)

        X = self._data.select_dtypes(include=[object])
        to_encode = X.columns
        print(to_encode)

        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)
        print(self._data.iloc[:, -1].value_counts())
        print(self._data)


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class ShopperData(DataLoader):

    def __init__(self, path='../data/online_shoppers_intention.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, sep=',')

    def data_name(self):
        return 'ShopperData'

    def class_column_name(self):
        return 'shopper'

    def _preprocess_data(self):
        print(self._data.columns)
        self._data.iloc[:, -1] = self._data.iloc[:, -1].astype(int)
        print(self._data)

        X = self._data.select_dtypes(include=[object])
        to_encode = X.columns
        print(to_encode)

        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)
        print(self._data.iloc[:, -1].value_counts())
        print(self._data.iloc[:, -1])


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

if __name__ == '__main__':
    # bank_data = BankMarketingData(verbose=True)
    # bank_data.load_and_process()
    # shop_data = ShopperData(verbose=True)
    # shop_data.load_and_process()

    ca_data = WhiteWineData(verbose=True)
    ca_data.load_and_process()

    ep_data = EpilepsyData(verbose=True)
    ep_data.load_and_process()

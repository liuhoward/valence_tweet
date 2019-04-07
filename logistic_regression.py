# -*- coding: utf-8 -*-
# 
# 3/28/19   Howard  create logistic_regression.py
#

from typing import Iterator, Iterable, Tuple, Text, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import matthews_corrcoef

NDArray = Union[np.ndarray, spmatrix]


def read_train(src_file: str):
    """Generates (id, tweet, dimension, score) tuples from the lines in an src file.

    :param src_file:
    :return:
    """
    id_list = list()
    tweet_list = list()
    # dimension_list = list()
    intensity_list = list()
    with open(src_file) as fp:
        reader = csv.DictReader(fp, delimiter='\t')
        for row in reader:
            record_id = row['ID']
            tweet = row['Tweet']
            dimension = row['Affect Dimension']
            if dimension != 'valence':
                print(record_id)
            intensity = (row['Intensity Class']).split(': ')[0]
            id_list.append(record_id)
            tweet_list.append(tweet)
            # dimension_list.append(dimension)
            intensity_list.append(intensity)

    return id_list, tweet_list, intensity_list


def read_test(src_file: str):
    """Generates (id, tweet, dimension, score) tuples from the lines in an src file.

    :param src_file:
    :return:
    """
    id_list = list()
    tweet_list = list()
    with open(src_file) as fp:
        reader = csv.DictReader(fp, delimiter='\t')
        for row in reader:
            record_id = row['ID']
            tweet = row['Tweet']
            dimension = row['Affect Dimension']
            if dimension != 'valence':
                print(record_id)
            id_list.append(record_id)
            tweet_list.append(tweet)

    return id_list, tweet_list


class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        # maximum ngram
        #self.ngram_max = 3
        # generate features based on texts
        # self.vectorizer = CountVectorizer(min_df=0, ngram_range=(1, self.ngram_max))
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(texts)

        # number of total features
        self.num_feature = len(self.vectorizer.vocabulary_)

    def index(self, feature: Text):
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        return self.vectorizer.vocabulary_[feature]

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """

        return self.vectorizer.transform(texts)


class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        # logistic regression model
        self.clf = LogisticRegression(random_state=0, solver='lbfgs')

    def train(self, features: NDArray, labels) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        # train model
        self.clf.fit(X=features, y=labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        # predict
        return self.clf.predict(X=features)

    def evaluate(self, features:NDArray, labels:NDArray) -> None:
        """evaluate performance on testing data"""

        print(self.clf.score(X=features, y=labels))
        y_pred = self.clf.predict(features)
        print(matthews_corrcoef(y_true=labels, y_pred=y_pred))


def main():

    data_path = 'data/'
    train_data = data_path + '2018-Valence-oc-En-train.txt'
    dev_data = data_path + '2018-Valence-oc-En-dev.txt'
    test_data = data_path + '2018-Valence-oc-En-test.txt'

    train_id, train_tweet, train_label = read_train(train_data)

    dev_id, dev_tweet, dev_label = read_train(dev_data)

    test_id, test_tweet = read_test(test_data)

    to_feature = TextToFeatures(train_tweet)

    cls = LogisticRegression(random_state=0, solver='lbfgs')
    cls.fit(X=to_feature(train_tweet), y=train_label)

    print(cls.score(X=to_feature(dev_tweet), y=dev_label))
    y_pred = cls.predict(to_feature(dev_tweet))
    print(matthews_corrcoef(y_true=dev_label, y_pred=y_pred))


if __name__ == "__main__":
    main()

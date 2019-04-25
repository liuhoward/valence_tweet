# -*- coding: utf-8 -*-
# 
# 4/24/19   Howard  create data.py
#

import numpy as np
import csv


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
    dimension_list = list()
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
            dimension_list.append(dimension)
            tweet_list.append(tweet)

    return id_list, tweet_list, dimension_list


def save_prediction(id_list, tweet_list, dimension_list, label_list, dst_file):
    """
        save predicted labels

    :param id_list:
    :param tweet_list:
    :param dimension_list:
    :param label_list:
    :param dst_file:
    :return:
    """

    head = 'ID\tTweet\tAffect Dimension\tIntensity Class\n'
    with open(dst_file, 'w') as fp:
        fp.write(head)
        for i in range(len(id_list)):
            fp.write('{}\t{}\t{}\t{}\n'.format(id_list[i], tweet_list[i], dimension_list[i], label_list[i]))


def gen_bert_predict():
    data_path = 'data/'
    test_data = data_path + '2018-Valence-oc-En-test.txt'
    pred_data = data_path + 'V-oc_en_pred.txt'

    test_id, test_tweet, test_dimesion = read_test(test_data)

    y_pred = list()
    bert_result_file = 'output/test_results.tsv'
    with open(bert_result_file, 'r') as fp:
        for line in fp:
            parts = line.replace('\n', '').split('\t')
            row = [float(x.strip()) for x in parts]
            label = np.argmax(row) - 3
            y_pred.append(label)

    save_prediction(test_id, test_tweet, test_dimesion, y_pred, pred_data)


def save_tweets(tweets, dst_file):
    with open(dst_file, 'w') as fp:
        for tweet in tweets:
            fp.write(tweet + '\n')


def generate_tweets():
    data_path = 'data/'
    train_data = data_path + '2018-Valence-oc-En-train.txt'
    dev_data = data_path + '2018-Valence-oc-En-dev.txt'
    test_data = data_path + '2018-Valence-oc-En-test.txt'

    train_id, train_tweet, train_label = read_train(train_data)

    dev_id, dev_tweet, dev_label = read_train(dev_data)

    test_id, test_tweet, test_dimesion = read_test(test_data)

    save_tweets(train_tweet, data_path + 'train_tweets.txt')
    save_tweets(dev_tweet, data_path + 'dev_tweets.txt')
    save_tweets(test_tweet, data_path + 'test_tweets.txt')


def main():

    # gen_bert_predict()
    generate_tweets()


if __name__ == "__main__":
    main()
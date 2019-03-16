import os
from collections import defaultdict

from util.token_util import TokenFilter

stop_words_dict = "../data/stopwords/stopwords_eng.txt"


def create_bag_of_words(text):
    """
    creates a case insensitive bag of words set with key(token_name) and value(frequency) per document
    :param text:
    :return:
    """
    bag_of_words = defaultdict(lambda: 0)
    token_filter = TokenFilter(stop_words_dict)

    tokens = text.lower().split()
    for token in tokens:
        filtered = token_filter.filter(token)
        if filtered is not None:
            bag_of_words[filtered] += 1

    return bag_of_words


def create_bows_from_path(path_name):
    """
    loads txt files from directory and breaks them down to unigram word features (bag of words)
    :param path_name:
    :return:
    """
    bow_list = []
    files = os.listdir(path_name)
    for f in files:
        with open(path_name+f, "r") as text_file:
            bow = create_bag_of_words(text_file.read())
            bow_list.append(bow)
    return bow_list

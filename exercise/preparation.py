from collections import defaultdict

stop_words_dict = "../data/stopwords/stopwords_eng.txt"


def create_bag_of_words(text):
    """
    creates a case insensitive bag of words set with key(token_name) and value(frequency) per document
    :param text:
    :return:
    """
    bag_of_words = defaultdict(lambda: 0)
    # TODO

    return bag_of_words


def create_bows_from_path(path_name):
    """
    loads txt files from directory and breaks them down to unigram word features (bag of words)
    :param path_name:
    :return:
    """
    bow_list = []
    # TODO

    return bow_list

#!/bin/python3
from sentiment_classifier import SentimentClassifier

from data import create_bows_from_path
from util.object_util import save_object

MODEL_PATH = "../data/model/"
TRAINING_DATA_PATH = "../data/training/"


# collect bag_of_words for positive samples
positive_bow_list = []
positive_training_path = TRAINING_DATA_PATH + "pos/"

positive_bow_list = create_bows_from_path(positive_training_path)

# collect bag_of_words for negative samples
negative_bow_list = []
negative_training_path = TRAINING_DATA_PATH + "neg/"

negative_bow_list = create_bows_from_path(negative_training_path)


# train the bayes classifier
classifier = SentimentClassifier()
classifier.train(positive_bow_list, negative_bow_list)

save_object(classifier, MODEL_PATH + "classifier.model")


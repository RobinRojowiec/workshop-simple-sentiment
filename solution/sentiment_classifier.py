import math
from collections import defaultdict


class SentimentClassifier:
    def __init__(self):
        self.classes: [] = ["positive", "negative"]
        self.file_name: str = '../data/naive_bayes.model'

        self.feature_probabilities: dict = defaultdict(self._default_tuple)
        self.class_probabilities: dict = {clazz: 0.5 for clazz in self.classes}

    def _default_tuple(self):
        """
        default tuple for feature probabilities
        :return:
        """
        return [0.0 for i in range(len(self.classes)+1)]

    def predict(self, bag_of_words):
        """
        predicts the probabilites per class based on unigram word features
        :param bag_of_words:
        :return:
        """
        prob_positive = math.log(self.class_probabilities["positive"])
        prob_negative = math.log(self.class_probabilities["negative"])

        for word in bag_of_words:
            if word in self.feature_probabilities:

                feature_prob_positive = self.get_feature_probability(word, "positive")
                prob_positive += math.log(feature_prob_positive)

                feature_prob_negative = self.get_feature_probability(word, "negative")
                prob_negative += math.log(feature_prob_negative)

        # normalize
        normalized_prob_positive = 1/(prob_positive / (prob_positive + prob_negative))
        normalized_prob_negative = 1/(prob_negative / (prob_positive + prob_negative))

        return [("positive", normalized_prob_positive), ("negative", normalized_prob_negative)]

    def predict_class(self, bag_of_words):
        """
        Returns the class with the highest probability
        :param bag_of_words:
        :return:
        """
        probabilites = self.predict(bag_of_words)
        probabilites.sort(key=lambda x: x[1], reverse=True)
        return probabilites[0][0]

    def get_feature_probability(self, feature, clazz):
        """
        Returns the probability of a feature given a class
        :param feature:
        :param clazz:
        :return:
        """
        index: int = self.class_to_index(clazz)
        probability: float = self.feature_probabilities[feature][index]
        return probability + self.feature_probabilities[feature][2]

    def class_to_index(self, clazz):
        """
        Converts a class name (string) to the appropriate index (int)
        :param clazz:
        :return:
        """
        return self.classes.index(clazz)

    def train(self, bow_positive: [], bow_negative: []):
        """
        calculates feature probabilities
        :param bow_positive:
        :param bow_negative:
        :return:
        """
        all_positive_token_count = self.sum_counts(bow_positive, 0)
        all_negative_token_count = self.sum_counts(bow_negative, 1)

        for token in self.feature_probabilities:
            pos, neg, _ = self.feature_probabilities[token]
            if pos > 0:
                self.feature_probabilities[token][0] = pos / (pos+neg)

            if neg > 0:
                self.feature_probabilities[token][1] = neg / (pos+neg)

            self.feature_probabilities[token][2] = (pos+neg) / (all_negative_token_count + all_positive_token_count)

    def sum_counts(self, bow_list, index):
        """
        sums up feature counts
        :param bow_list:
        :param index:
        :return:
        """
        overall_count: int = 0
        for bow in bow_list:
            for token in bow:
                self.feature_probabilities[token][index] += bow[token]
                overall_count += bow[token]
        return overall_count

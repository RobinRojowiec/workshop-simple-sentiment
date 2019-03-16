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
        # TODO

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
        # TODO

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
            # TODO
            pass


    def sum_counts(self, bow_list, index):
        """
        sums up feature counts
        :param bow_list:
        :param index:
        :return:
        """
        overall_count: int = 0
        # TODO

        return overall_count

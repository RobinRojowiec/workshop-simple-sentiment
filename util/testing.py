import os
import pickle

from util.confusion_matrix import ConfusionMatrix
from util.token_util import TokenFilter, Tokenizer, create_bag_of_words

main_path = "data/test/"
pos_path = main_path + "pos"
neg_path = main_path + "neg"
paths = [[pos_path, "positive"], [neg_path, "negative"]]


def test_model(model_path: str, stop_word_file="data\stopwords_eng.txt"):
    print("starting testing")

    token_filter = TokenFilter(stop_word_file)
    tokenizer = Tokenizer()

    with open(model_path, "rb") as fb:
        bayes = pickle.load(fb)
    cf = ConfusionMatrix(["positive", "negative"])

    for path in paths:
        for file in os.listdir(path[0]):
            with open(path[0] + "/" + file) as f:
                tokens = tokenizer.tokenize(f)
                bag_of_words = create_bag_of_words(token_filter, tokens, True)

                predicted_class = bayes.predict_class(bag_of_words)
                print("Predicted: "+predicted_class+", Real: "+path[1])

                cf.add_prediction(path[1], predicted_class)

    print(cf)
    print("Accuracy: "+str(cf.accuracy_average()))
    print("Recall: "+str(cf.recall_average()))
    print("Precision: "+str(cf.precision_average()))
    print("F-Measure: "+str(cf.f_measure_average()))


if __name__ == '__main__':
    test_model('data/naive_bayes.model')
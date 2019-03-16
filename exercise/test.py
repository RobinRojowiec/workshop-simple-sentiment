from sentiment_classifier import SentimentClassifier

from data import create_bows_from_path
from util.confusion_matrix import ConfusionMatrix
from util.object_util import load_object

MODEL_PATH = "../data/model/"
TEST_DATA_PATH = "../data/test/"
# load model
bayes: SentimentClassifier = load_object(MODEL_PATH + "classifier.model")
cf: ConfusionMatrix = ConfusionMatrix(bayes.classes)

# load bows
# collect bag_of_words for positive samples
positive_bow_list = []
positive_test_path = TEST_DATA_PATH + "pos/"

positive_bow_list = create_bows_from_path(positive_test_path)

for bow in positive_bow_list:
    clazz: str = bayes.predict_class(bow)
    cf.add_prediction("positive", clazz)


negative_bow_list = []
negative_test_path = TEST_DATA_PATH + "neg/"

negative_bow_list = create_bows_from_path(negative_test_path)

for bow in negative_bow_list:
    clazz: str = bayes.predict_class(bow)
    cf.add_prediction("negative", clazz)

print(cf)
print("Accuracy: "+str(round(cf.accuracy_average()*100))+"%")

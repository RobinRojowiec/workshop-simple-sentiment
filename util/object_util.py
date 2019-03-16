import os
import pickle


def save_object(obj, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w+', encoding="utf-8"):
            pass

    with open(file_name, 'wb') as output:
        pickle.dump(obj, output)


def load_object(file_name):
    with open(file_name, 'rb') as input:
        return pickle.load(input)

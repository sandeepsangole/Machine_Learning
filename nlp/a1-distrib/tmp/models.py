# models.py

from sentiment_data import *
from utils import *
import numpy as np
import math
from collections import Counter
import random

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = [
            "a", "an", "and", "the", "in", "of", "to", "for", "with", "on",
            "by", "as", "at", "but", "or", "is", "it", "he", "she", "I", "you",
            "we", "they", "me", "him", "her", "my", "your", "our", "their",
            "this", "that", "these", "those", "be", "am", "are", "was", "were",
            "do", "does", "did", "have", "has", "had", "can", "could", "will",
            "would", "should"
        ]

    def get_indices_for_words(self, sentence: List[str]):
        for word in sentence:
            if word.lower() not in self.stop_words:
                self.indexer.add_and_get_index(word.lower())

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        feature_vector = Counter()

        for word in sentence:
            if self.indexer.contains(word.lower()):
                index = self.indexer.index_of(word.lower())
                feature_vector.update([index])
        return feature_vector



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = [
            "a", "an", "and", "the", "in", "of", "to", "for", "with", "on",
            "by", "as", "at", "but", "or", "is", "it", "he", "she", "I", "you",
            "we", "they", "me", "him", "her", "my", "your", "our", "their",
            "this", "that", "these", "those", "be", "am", "are", "was", "were",
            "do", "does", "did", "have", "has", "had", "can", "could", "will",
            "would", "should"
        ]

    def get_indices_for_words(self, sentence: List[str]):
        for count in range(len(sentence) - 1):
            bigram = sentence[count] +' '+sentence[count+1]
            self.indexer.add_and_get_index(bigram.lower())

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        feature_vector = Counter()
        for count in range(len(sentence) - 1):
            bigram = sentence[count]+' '+sentence[count+1]
            if self.indexer.contains(bigram.lower()):
                index = self.indexer.index_of(bigram.lower())
                feature_vector.update([index])
        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = [
            "a", "an", "and", "the", "in", "of", "to", "for", "with", "on",
            "by", "as", "at", "but", "or", "is", "it", "he", "she", "I", "you",
            "we", "they", "me", "him", "her", "my", "your", "our", "their",
            "this", "that", "these", "those", "be", "am", "are", "was", "were",
            "do", "does", "did", "have", "has", "had", "can", "could", "will",
            "would", "should"
        ]
    def get_indexer(self):
        return self.indexer

    def get_indices_for_words(self, sentence: List[str]):
        for word in sentence:
            if word.lower() not in self.stop_words:
                self.indexer.add_and_get_index(word.lower())

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()

        for word in sentence:
            if self.indexer.contains(word.lower()):
                index = self.indexer.index_of(word.lower())
                feature_vector.update([index])

        return feature_vector


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.indexer = self.feature_extractor.get_indexer()
        self.bag_size = self.indexer.__len__()
        self.weights = np.zeros((self.bag_size), dtype = int)

    def update_weight(self, y, y_pred, lr, sentence:List[str]):
        feature_vector = self.featurizer(sentence)
        for index in feature_vector.keys():
            val = feature_vector[index]
            self.weights[index] += (y - y_pred) * lr * val

    def featurizer(self, sentence: List[str]):
        return self.feature_extractor.extract_features(sentence)

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.featurizer(sentence)

        prod = 0
        for index in feature_vector.keys():
            val = feature_vector[index]
            prod += self.weights[index] * val

        if prod > 0:
            y_pred = 1
        else:
            y_pred = 0

        return y_pred



class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.indexer = self.feature_extractor.get_indexer()
        self.bag_size = self.indexer.__len__()
        self.weights = np.zeros((self.bag_size), dtype = int)

    def sigmoid(self, x):
        res= 1 / (1 + math.exp(-x))
        # res = max(res, 1e-8)
        # res = min(res, 1 - 1e-8)
        return res

    def featurizer(self, sentence: List[str]):
        return self.feature_extractor.extract_features(sentence)

    def update_weight(self, y, lr, sentence: List[str]):
        feature_vector = self.featurizer(sentence)

        prod = 0
        for index in feature_vector.keys():
            val = feature_vector[index]
            prod += self.weights[index] * val

        y_pred = self.sigmoid(prod)

        for index in feature_vector.keys():
            val = feature_vector[index]
            # self.weights[index] = self.weights[index] - lr * (-y * val * (1 - y_pred) + (1 - y) * val * y_pred)
            if y == 1:
                self.weights[index] += lr * val * (1 - y_pred)
            else:
                self.weights[index] -= lr * val * y_pred

    def featurizer(self, sentence: List[str]):
        return self.feature_extractor.extract_features(sentence)

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.featurizer(sentence)

        prod = 0
        for index in feature_vector.keys():
            val = feature_vector[index]
            prod += self.weights[index] * val

        pred = self.sigmoid(prod)

        if pred > 0.5:
            y_pred = 1
        else:
            y_pred = 0

        return y_pred


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    for example in train_exs:
        sentence = example.words
        feat_extractor.get_indices_for_words(sentence)

    epochs = 20
    model = PerceptronClassifier(feat_extractor)

    lr = 1.0
    for epoch in range(epochs):
        for example in train_exs:
            y = example.label
            sentence = example.words
            y_pred = model.predict(sentence)
            model.update_weight(y, y_pred, lr, sentence)
        random.shuffle(train_exs)

    return model

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for example in train_exs:
        sentence = example.words
        feat_extractor.get_indices_for_words(sentence)

    epochs = 30
    model = LogisticRegressionClassifier(feat_extractor)

    lr = 12.0
    for epoch in range(epochs):
        random.shuffle(train_exs)
        for example in train_exs:
            y = example.label
            sentence = example.words
            y_pred = model.predict(sentence)
            model.update_weight(y, lr, sentence)
        lr = lr - (1/epochs)

    return model

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
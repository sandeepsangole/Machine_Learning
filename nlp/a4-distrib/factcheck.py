# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.metrics import jaccard_distance
from nltk.stem import PorterStemmer
import re
class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # Calculate probabilities from logits using softmax
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        # Determine the predicted label based on the highest probability
        predicted_label = ["entailment", "neutral", "contradiction"][probabilities.index(max(probabilities))]

        # raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something
        return predicted_label, probabilities


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:



        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm")
        self.token_weights = None

    def jaccard_similarity(self, str1, str2):
        a = set(ngrams(str1.split(),2))
        b = set(ngrams(str2.split(),2))
        intersection = len(a.intersection(b))
        union = len(a) + len(b) - intersection

        if union == 0:
            return 0
        return intersection / union

    def weighted_jaccard_similarity(self, fact, passage):
        fact_bigrams = set(fact.split())
        passage_bigrams = set(passage.split())

        if len(fact_bigrams) == 0:
            return 0

        weighted_intersection = fact_bigrams.intersection(passage_bigrams)
        fact_weight = len(weighted_intersection) / len(fact_bigrams)


        similarity_score = fact_weight * (len(weighted_intersection) / len(fact_bigrams.union(passage_bigrams)))

        return similarity_score

    def stem_text(self, text):
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    def preprocess_text(self, text):
        # Load Spacy model and preprocess text
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc if token.is_alpha]
        words = [word for word in words if word not in self.stop_words]
        return " ".join(words)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact1 = self.preprocess_text(fact)
        sentence_tokens = self.stem_text(fact1)
        for passage in passages:
            for key, value in passage.items():
                if key == 'text':
                    pass_text = self.preprocess_text(value)
                    pass_text = self.stem_text(pass_text)
                    cleaned_text = re.sub(r'<s>.*?</s>', '', pass_text)
                    jaccard_similarity = self.weighted_jaccard_similarity(sentence_tokens, cleaned_text)
                    # print("Jaccard Similarity:", jaccard_similarity, "for ", value)
                    if jaccard_similarity > 0.0175:
                        return 'S'
        return 'NS'


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.nlp = spacy.load("en_core_web_sm")

    def predict(self, fact: str, passages: List[dict]) -> str:
        # raise Exception("Implement me")
        # fact_sentences = fact.split(" ")
        # Loop over the passages and sentences, and check for entailment.
        for passage in passages:
            for key, value in passage.items():
                if key == 'text':

                    doc = self.nlp(value)
                    sentences = [sent.text for sent in doc.sents]

                    for sentence in sentences:
                        entailment_prediction, score = self.ent_model.check_entailment(premise=sentence, hypothesis=fact)

                        if entailment_prediction == "entailment":
                            return "S"  # Supported

        # If the model does not predict entailment for any sentence in any passage, then the fact is not supported.
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

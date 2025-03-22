# MultipleFiles/nltk_utils.py
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer

# Initialize stemmer and context encoder
stemmer = PorterStemmer()
context_encoder = MultiLabelBinarizer()

def tokenize(sentence):
    """Convert sentence to lowercase word tokens"""
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    """Normalize and stem words"""
    return stemmer.stem(word.lower().strip())

def bag_of_words(tokenized_sentence, all_words):
    """Create BoW vector with stemming"""
    stemmed_sentence = [stem(w) for w in tokenized_sentence]
    return np.array([1 if w in stemmed_sentence else 0 for w in all_words], dtype=np.float32)

def context_vector(current_context, all_contexts):
    """Create context feature vector"""
    return np.array([1 if ctx in current_context else 0 for ctx in all_contexts], dtype=np.float32)

def prepare_full_input(bow_vector, context_vector):
    """Combine BoW and context features"""
    return np.concatenate((bow_vector, context_vector))
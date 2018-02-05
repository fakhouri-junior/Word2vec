import tensorflow as tf
import numpy as np
from learn_word2vec.Skip_gram_model import add_dictionaries_to_index_words
import os.path
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt

# default values, can be changed by running program from command line
TOKENIZED_INPUT_FILE = "tokenized_file.txt"
TRAIN_DATA_FILE = "word_pairs.txt"



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
EPOCHS = 100
NUMBER_OF_TRAINING_EXAMPLES = 68  # according to my word_pairs.txt toy example
word2int, int2word = add_dictionaries_to_index_words.make_word2int_and_int2word(TOKENIZED_INPUT_FILE)

VOCAB_LENGTH = len(word2int.keys())
HIDDEN_DIM = 25  # play with this as you wish, i choose 25 because corpus is very tiny

START_LEARNING_RATE = 0.0001

"""
Create a generator that can read a file and pass x,y values,
this generator will be used by Dataset Tensorflow to get values
"""


def generate_sample_training():
    """
    :yield: x,y hot vectors for each line of word_pairs
    """
    # boilerplate code that runs only one time, reading the file for ex
    if os.path.isfile(TRAIN_DATA_FILE):
        my_file = open(TRAIN_DATA_FILE, 'r')
        for line in my_file:
            words = line.split(',')
            input_word = words[0]
            output_word = words[1]
            input_word = input_word.strip()
            output_word = output_word.strip()
            yield to_one_hot(input_word, word2int), to_one_hot(output_word, word2int)

    else:
        raise ValueError("File does not exist")
    pass


def to_one_hot(word, word2int):
    # determine vocab_size (vocabulary size, how many unique words we have)
    vocab_size = len(word2int.keys())
    if word in word2int.keys():
        index = word2int[word]
        one_hot_vector = np.zeros(vocab_size, dtype=np.float32)
        one_hot_vector[index] = 1
        return one_hot_vector
    else:
        raise ValueError("This word does not exist in vocabulary {}".format(word))


# create the dataset and transform it
dataset = tf.data.Dataset.from_generator(generate_sample_training, output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([None]), tf.TensorShape([None])))
dataset = dataset.batch(512)

# create one iterator and use if with different dataset, ideal way to work with train_data, and test_Data
# but here we only have one data_Set (train)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
X, Y = iterator.get_next()
# create iterator initialization op with train dataset
iterator_init_op_train = iterator.make_initializer(dataset)

"""
similarily if you have test_Dataset iterator_init_op_test = iterator.make_initializer(test_Dataset)
and just run that op when you need to use test data, usually after training is done
"""

"""
Building Neural Net with 1 hidden layer (100 dim)
"""
W1 = tf.get_variable(name="weights_1", shape=[VOCAB_LENGTH, HIDDEN_DIM], dtype=tf.float32,
                     initializer=tf.truncated_normal_initializer())
b1 = tf.get_variable(name="bias_1", shape=[1, HIDDEN_DIM], dtype=tf.float32,
                     initializer=tf.constant_initializer(value=0.0))
W2 = tf.get_variable(name="weights_2", shape=[HIDDEN_DIM, VOCAB_LENGTH], dtype=tf.float32,
                     initializer=tf.truncated_normal_initializer())
b2 = tf.get_variable(name="bias_2", shape=[1, VOCAB_LENGTH], dtype=tf.float32,
                     initializer=tf.constant_initializer(value=0.0))

"""
Add histogram visualization for variables in tensorboard
"""
tf.summary.histogram("W1_Histogram", W1)
tf.summary.histogram("W2_Histogram", W2)

# note: we don't apply activation function, W1 + b1 is our hidden representation of word_Vectors if you will
Z1 = tf.matmul(X, W1) + b1
Z2 = tf.matmul(Z1, W2) + b2
y_hat = tf.nn.softmax(Z2)

# loss, we will use cross entropy
loss = tf.reduce_mean(- tf.reduce_sum(Y * tf.log(y_hat), reduction_indices=[1]))
"""
Add learning rate decay
use the equation:
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
                        
    Note: if staircase = True the division will result in integer
    here we don't want that, because if it is an int  the learning rate 
    will only decay if global_step matches the decay steps 
"""
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step=global_step,
                                           decay_steps=100000, decay_rate=0.96)

# Optimizer Adam, play with learning rate till you find best value
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

"""
Create TensorBoard FileWriter , variables initializer and Saver to save variables
"""
writer = tf.summary.FileWriter(logdir="events_board", graph=tf.get_default_graph())
saver = tf.train.Saver()
var_init = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialize all variables
    sess.run(var_init)

    for i in range(EPOCHS):
        # run iterator initializer op with train dataset every epoch
        sess.run(iterator_init_op_train)
        total_loss = 0
        try:
            while True:
                merge = tf.summary.merge_all()

                _, l, summary= sess.run([train_op, loss, merge])
                total_loss += l

                writer.add_summary(summary, sess.run(global_step))
        except tf.errors.OutOfRangeError:
            pass

        print("Epoch {0} is done and total loss is {1}".format(i, total_loss / NUMBER_OF_TRAINING_EXAMPLES))

        # handle saving every 10 epochs
        if i % 5 == 0:
            # training is done, save and close writer
            save_path = saver.save(sess, "checkpoints\\", global_step=global_step)
    # training is done, save and close writer
    save_path = saver.save(sess, "checkpoints\\", global_step=global_step)
    writer.close()

    # OUR Word2vec
    vectors = sess.run(W1 + b1)

# visualize
model = TSNE(n_components=2, random_state=0)
vectors = model.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(len(int2word.keys())):
    plt.scatter(*vectors[idx, :], color='steelblue')
    plt.annotate(int2word[idx], (vectors[idx, 0], vectors[idx, 1]), alpha=0.7)

plt.show()

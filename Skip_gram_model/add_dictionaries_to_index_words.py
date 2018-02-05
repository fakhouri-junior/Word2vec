import os.path
from collections import Counter

def get_number_training_examples(file_name="word_pairs.txt"):
    my_file = open(file_name, 'r')
    lines = my_file.readlines()
    print (len(lines))
    return lines


#from string
def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    words = words.split()
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int2word = {ii: word for ii, word in enumerate(sorted_vocab)}
    word2int = {word: ii for ii, word in int2word.items()}

    return word2int, int2word

# from file, utilizes create_lookup_tables
def make_word2int_and_int2word(preprocessed_corpus_file):
    # make sure file exists
    if os.path.isfile(preprocessed_corpus_file):

        my_file = open(preprocessed_corpus_file, 'r')
        all_text = my_file.read()
         # get rid of double spaces if any
        all_text = all_text.replace('  ', ' ')

        word2int, int2word = create_lookup_tables(all_text)
        my_file.close()
        return word2int, int2word
    else:
        raise ValueError("File does not exist")

#
# word2int, int2word = make_word2int_and_int2word(preprocessed_corpus_file='tokenized_file.txt')
# print(word2int)
# print(int2word)

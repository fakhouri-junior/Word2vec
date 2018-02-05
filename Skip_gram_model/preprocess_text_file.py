import os.path
import ntpath
import argparse
from collections import Counter
import generate_file_training_data_word_pairs
import re

INPUT_FILE = "corpus.txt" #default
OUTPUT_FILE = "word_pairs.txt" #default



def replace_function_eos(pattern):
    m = pattern.group(0)
    res = m[0] + " <EOS> " + m[2]
    return res

def find_and_insert_end_of_sentence_token(long_string):
    """
    2 cases we search for .\n (handled by preprocess) and . Capital_Letter
    :param long_string: any string
    :return: same string with <EOS> token inserted where end of sentence is identified
    """
    long_string = re.sub(r'[.] [A-Z]', replace_function_eos, long_string)
    return long_string


# to be used for word<PERIOD>
def replace_function_space(pattern):
     m = pattern.group(0)
     index_of_angle_bracket = m.find("<")
     res = m[:index_of_angle_bracket] + ' ' + m[index_of_angle_bracket:]
     return res


def find_word_period_combination_and_insert_space(long_string):
    long_string = re.sub(r'\w+<PERIOD> ', replace_function_space, long_string)
    return long_string


def preprocess(filename_path):
    """
    Tokenize punctuation and delete words which appear less than 5 times
    generate word2int, int2word from unique words

    :param filename_path: path to the filename needs to be adjusted
    :return: a list of filtered words
    """

    if os.path.isfile(filename_path):
        # open the file and read it line by line
        my_file = open(filename_path, 'r')
        all_text = my_file.read()

        # todo: delete rare words which appear only 5 times or less

        # identify end of sentences and replace with <EOS> token
        all_text = all_text.replace('.\n', '. <EOS> \n')
        all_text = find_and_insert_end_of_sentence_token(all_text)

        all_text = all_text.replace('.', '<PERIOD>')
        all_text = all_text.replace(',', ' <COMMA> ')
        all_text = all_text.replace('"', ' <QUOTATION_MARK> ')
        all_text = all_text.replace(';', ' <SEMICOLON> ')
        all_text = all_text.replace('!', ' <EXCLAMATION_MARK> ')
        all_text = all_text.replace('?', ' <QUESTION_MARK> ')
        all_text = all_text.replace('(', ' <LEFT_PAREN> ')
        all_text = all_text.replace(')', ' <RIGHT_PAREN> ')
        all_text = all_text.replace('--', ' <HYPHENS> ')
        all_text = all_text.replace('?', ' <QUESTION_MARK> ')
        all_text = all_text.replace(':', ' <COLON> ')

        #insert space between word<PERIOD>
        all_text = find_word_period_combination_and_insert_space(all_text)
        # eliminate double spaces
        all_text = all_text.replace('  ', ' ')

        # generate tokenized file for reference and manual checking if necessary
        tokenized_file = open("tokenized_file.txt",'w')
        tokenized_file.write(all_text)
        tokenized_file.close()


        # this is to generate training data
        sentences = all_text.split('<EOS>')

        my_file.close()
        return sentences

    else:
        # throw error
        raise ValueError("filename does not exist")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default="corpus.txt", dest="input_file", type=str, help="Path to input file")
    parser.add_argument('-o', default="word_pairs.txt", dest="output_file", type=str, help="Path to output file, this will be training data")
    parser.add_argument('-w', default=5, dest="window_size", type=int, help="Window size value")
    args = parser.parse_args()
    sentences = preprocess(args.input_file)
    generate_file_training_data_word_pairs.generate_word_pairs(sentences, window_size=args.window_size, output_file=args.output_file)


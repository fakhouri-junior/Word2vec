import os.path

def make_word2int_and_int2word(preprocessed_corpus_file):
    # make sure file exists
    if os.path.isfile(corpus_file):
        # find all unique words
        all_words = []
        word2int = {}
        int2word = {}
        my_file = open(preprocessed_corpus_file, 'r')
        for line in my_file:
            line = line.strip()
            words_in_each_line = line.split(' ')
            for w in words_in_each_line:
                all_words.append(w)
        unique_words = set(all_words)

        # index each word
        for index, word in enumerate(unique_words):
            word2int[word] = index
            int2word[index] = word
        return word2int, int2word
    else:
        raise ValueError("File does not exist")


word2int, int2word = make_word2int_and_int2word(preprocessed_corpus_file='preprocessed.txt')
print(word2int)
print(int2word)

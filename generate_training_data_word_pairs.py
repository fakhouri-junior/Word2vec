import os.path


def generate_word_pairs(preprocessed_file, window_size=5, output_file="word_pairs.txt"):
    """
    our generated data will be a list of list, where each list is 2 words
    the current one and all possible neighbors
    [['he', 'is'],
     ['he', 'the'],
     ['is', 'he'],
     ['is', 'the'],
     ['is', 'king'],
     ['the', 'he'],
     ['the', 'is'],
     .......
        """
    data = []
    if os.path.isfile(preprocessed_file):
        my_file = open(preprocessed_file, 'r')
        for line in my_file:
            line = line.strip()
            sentence = line.split(' ')
            for word_index, word in enumerate(sentence):
                for nb_word in sentence[max(word_index - window_size, 0) : min(word_index + window_size, len(sentence)) + 1]:
                    if nb_word != word:
                        data.append([word, nb_word])

        write_file = open(output_file, 'w')
        for word_pair in data:
            write_file.write(word_pair[0] +","+ word_pair[1] +"\n")

    else:
        raise ValueError("File does not exist")


# test
generate_word_pairs("preprocessed.txt", window_size=5, output_file="word_pairs.txt")

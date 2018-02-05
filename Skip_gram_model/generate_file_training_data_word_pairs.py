import os.path


def generate_word_pairs(sentences, window_size=5, output_file="word_pairs.txt"):
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
     output file looks like:
        up,<PERIOD>
        my,the
        my,boss
        my,<PERIOD>
        my,what's
        my,up
        """
    data = []
    for sentence in sentences:
        sentence = sentence.split()
        for word_index, word in enumerate(sentence):
            for nieghbour_word in sentence[max(word_index - window_size, 0): min(word_index + window_size, len(sentence)) + 1]:
                if nieghbour_word != word:
                    data.append([word, nieghbour_word])


    write_file = open(output_file, 'w')
    for word_pair in data:
        write_file.write(word_pair[0] + "," + word_pair[1] + "\n")

# test
generate_word_pairs(["He", "The", "King", "Queen"], window_size=5, output_file="word_pairs.txt")

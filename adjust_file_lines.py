import os.path
import ntpath
import argparse

INPUT_FILE = "corpus.txt" #default
OUTPUT_FILE = "preprocessed.txt" # default

def adjust_file(filename_path, output_file_name):
    """
    write each sentence in one line

    :param filename_path: path to the filename needs to be adjusted
    :return: a new filename where each line is one sentence
    """
    # make sure file name path is correct
    lines = []
    if os.path.isfile(filename_path):
        # open the file and read it line by line
        my_file = open(filename_path, 'r')
        for line in my_file:
            # replace every dot with new line
            line = line.replace('.', ' \n')
            # split on \n
            line = line.split('\n')
            for sub_line in line:
                sub_line = sub_line.strip()
                # ignore empty lines
                if sub_line == "":
                    continue
                lines.append(sub_line)
        # write a new file
        new_file = open(output_file_name, "w")
        for l in lines:
            new_file.write(l + '\n')

    else:
        # throw error
        raise ValueError("filename does not exist")


# adjust_file("corpus.txt")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default="corpus.txt", dest="input_file", type=str, help="Path to input file")
    parser.add_argument('-o', default="preprocessed.txt", dest="output_file", type=str, help="name of output file")
    args = parser.parse_args()
    adjust_file(args.input_file, args.output_file)


import os.path
import ntpath

def adjust_file(filename_path):
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
            # print(line)
            # split on \n
            line = line.split('\n')
            for sub_line in line:
                sub_line = sub_line.strip()
                # ignore empty lines
                if sub_line == "":
                    continue
                lines.append(sub_line)
        # print(lines)
        # write a new file
        new_file = open(ntpath.basename(filename_path)+"_preprocessed.txt", "w")
        for l in lines:
            new_file.write(l + '\n')

    else:
        # throw error
        raise ValueError("filename does not exist")


adjust_file("corpus.txt")

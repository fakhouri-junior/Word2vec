# Word2vec
Word2vec implementation in Python from scratch using Skip-gram model .... " learning word embeddings representation "

If you are familiar with Word2vec and you would like to see full implementation from scratch then this repository is right for you,
otherwise I would recommened reading this blog post by the awesome Chris McCormick:
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Then come back here and go through the files and you will be able to follow along.

## Usage

Run preprocess_text_file.py with input file path, and output file name you would like to get for output data
```
python preprocess_text_file.py -i input_file.txt -o name_of_output_file.txt
```

or if you would like to use the default files and run the model on a really tiny corpus just to experiment with code rather than get results
then just run the file

python preprocess_text_file.py

the output file path will contain the training data (Word_pairs or whatever name specified for output_file) and another file will be created
called tokenized_file.txt, this will be used by the model to establish a dictionary mapping
from each word to an integer.


Finally run train_model_using_dataset_input_pipeline.py to start training


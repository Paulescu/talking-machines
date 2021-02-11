# Steps

1. Pre-process original datasets:
    ```
    train_none_original.txt -> train_none_original.csv
    val_none_original.txt   -> train_none_original.csv
    test_none_original.txt  -> train_none_original.csv
    ```
2. Build vocabulary from pre-processed dataset, using torchtext Field, 
TabularDataset APIs

    * Use the TreebankWordTokenizer from nltk.
    * Use glove 300-dimensional


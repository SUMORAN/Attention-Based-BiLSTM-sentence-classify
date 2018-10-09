import numpy as np
import pandas as pd
import nltk
import re


train_df = pd.read_csv("data/train_text.csv")
test_df = pd.read_csv("data/test_text.csv")

train_sentence_length = max([len(nltk.word_tokenize(x)) for x in train_df['sentence']])
test_sentence_length = max([len(nltk.word_tokenize(x)) for x in test_df['sentence']])
MAX_SENTENCE_LENGTH = max(train_sentence_length, test_sentence_length)

# LABELS_COUNT = 19
LABELS_COUNT = 2

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    This is for splitting English, changing all word to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(path):
    # read training data from CSV file
    df = pd.read_csv(path)

    # Text data
    # cause chanyelian data are chinese, and there are no space between sentence and label
    # 通过 为 上游 和 中间 贸易商 提供 供应链 金融 服务 解决 其 资金 短缺 问题 0
    # => sentence = 通过 为 上游 和 中间 贸易商 提供 供应链 金融 服务 解决 其 资金 短缺 问题
    #     label = 0
    # =>  0  => [1 0]
    #     1  => [0 1]
    #
    x_text = df['sentence'].tolist()
    sentence = []
    label = []
    for s in x_text:
        sen = s.split()
        length = len(sen)
        label_temp = sen[-1]
        # print(y_temp)
        if label_temp == '0':
            label_temp = [1, 0]
        elif label_temp == '1':
            label_temp = [0, 1]
        label.append(label_temp)
        sentence.append(' '.join(sen[:length-1]))

    # Label Data
    # y = df['label']
    # labels_flat = y.values.ravel()
    #
    # labels_count = np.unique(labels_flat).shape[0]
    #
    # # convert class labels from scalars to one-hot vectors
    # # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # # ...
    # # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    # def dense_to_one_hot(labels_dense, num_classes):
    #     num_labels = labels_dense.shape[0]
    #     index_offset = np.arange(num_labels) * num_classes
    #     labels_one_hot = np.zeros((num_labels, num_classes))
    #     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #     return labels_one_hot
    #
    # labels = dense_to_one_hot(labels_flat, labels_count)
    # labels = labels.astype(np.uint8)
    # y = df['label'].tolist()
    # sentence = np.array(sentence)
    # y = np.array(y)
    # print(y)
    return sentence, label


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("Total {} epochs".format(num_epochs))
    print("{} steps for each epoch".format(num_batches_per_epoch))
    print("==========")
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        print('\033[1;32mepoch {}: \033[0m'.format(epoch))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



if __name__ == "__main__":
    print("Train / Test file created")
    #
    # load_data_and_labels("data/test_google.csv")
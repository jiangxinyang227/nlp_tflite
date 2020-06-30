
import os
import json
import random
from collections import Counter
from itertools import chain


class DataHelper(object):
    """
    
    """

    def __init__(self, train_file, eval_file, output_path, max_len):
        """
        
        """
        self.train_file = train_file
        self.eval_file = eval_file
        self.output_path = output_path
        self.max_len = max_len

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def read_file(self, file_path):
        """
        """
        data = []
        with open(file_path, "r") as fr:
            for line in fr.readlines():
                line = line.replace("\n", "<eos>")
                data.append(line.split())
        return data

    def padding(self, data, word_to_index, label_to_index):
        """

        """
        data = [tokens[:self.max_len + 1] for tokens in data]
        inputs = [tokens[:-1] for tokens in data]
        labels = [tokens[1:] for tokens in data]

        input_ids = [[word_to_index.get(token, word_to_index["<unk>"]) for token in tokens] for tokens in inputs]
        label_ids = [[label_to_index.get(token, label_to_index["<unk>"]) for token in tokens] for tokens in labels]

        input_pad = [tokens + [0] * (self.max_len - len(tokens)) if len(tokens) < self.max_len else tokens
                     for tokens in input_ids]

        return list(zip(input_pad, label_ids))

    def gen_data(self):
        """
        """
        train_data = self.read_file(self.train_file)
        eval_data = self.read_file(self.eval_file)

        counter = Counter(list(chain(*train_data)))
        sort_count = sorted(counter.items(), key=lambda x: (x[1], x[0]))
        words = [item[0] for item in sort_count]
        label_to_index = dict(zip(words, range(len(words))))
        word_to_index = dict(zip(["<pad>"] + words, range(len(words) + 1)))

        with open(os.path.join(self.output_path, "word_to_index.txt"), "w") as fw:
            for word, index in word_to_index.items():
                fw.write(word + "\t" + str(index) + "\n")

        with open(os.path.join(self.output_path, "label_to_index.txt"), "w") as fw:
            for label, index in label_to_index.items():
                fw.write(label + "\t" + str(index) + "\n")

        train_ids = self.padding(train_data, word_to_index, label_to_index)
        eval_ids = self.padding(eval_data, word_to_index, label_to_index)

        return train_ids, eval_ids

    @staticmethod
    def next_batch(data, batch_size):
        """
        
        """
        random.shuffle(data)
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            batch = data[i * batch_size: (i + 1) * batch_size]
            inputs = []
            labels = []
            sequence_lens = []
            for item in batch:
                inputs.append(item[0])
                labels.extend(item[1])
                sequence_lens.append(len(item[1]))

            yield dict(inputs=inputs, labels=labels, sequence_lens=sequence_lens)


from itertools import chain
import tensorflow as tf

from config import Config

from model import CharRnnModel


def padding(data, word_to_index, label_to_index, max_len):
    """

    """
    data = [tokens[:max_len + 1] for tokens in data]
    inputs = [tokens[:-1] for tokens in data]
    labels = [tokens[1:] for tokens in data]

    input_ids = [[word_to_index.get(token, word_to_index["<unk>"]) for token in tokens] for tokens in inputs]
    label_ids = [[label_to_index.get(token, label_to_index["<unk>"]) for token in tokens] for tokens in labels]

    input_pad = [tokens + [0] * (max_len - len(tokens)) if len(tokens) < max_len else tokens
                 for tokens in input_ids]

    return list(zip(input_pad, label_ids))


# 1, 加载词和标签索引表
label_to_index = {}
with open("output/label_to_index.txt", "r") as fr:
    for line in fr:
        label, index = line.strip().split("\t")
        label_to_index[label] = int(index)

index_to_label = {value: key for key, value in label_to_index.items()}
label_ids_list = list(label_to_index.values())

word_to_index = {}
with open("output/word_to_index.txt", "r") as fr:
    for line in fr:
        word, index = line.strip().split("\t")
        word_to_index[word] = int(index)

test_data = []
with open("data/ptb.test.txt", "r") as fr:
    for line in fr.readlines():
        line = line.replace("\n", "<eos>")
        test_data.append(line.split())

test_data_ids = padding(test_data, word_to_index, label_to_index, Config.max_sequence_length)

model = CharRnnModel(Config, False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# checkpoint_file = tf.train.latest_checkpoint("ckpt_model")
# model.saver.restore(sess, checkpoint_file)

inputs = model.inputs
predictions = model.predictions

pred_output = []
true_output = []
for item in test_data_ids:
    feed_dict = {
        inputs: [item[0]]
    }
    predictions_ = sess.run(predictions, feed_dict=feed_dict)
    true_output.append(item[1])
    pred_output.append(predictions_[:len(item[1])])

correct = 0
total = 0

for pred, true_ in zip(list(chain(*pred_output)), list(chain(*true_output))):
    total += 1
    if pred == true_:
        correct += 1

print("acc: ", correct / total)
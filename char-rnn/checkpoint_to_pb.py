import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from config import Config

from model import CharRnnModel

model = CharRnnModel(Config, False)

sess = tf.Session()
checkpoint_file = tf.train.latest_checkpoint("ckpt_model")
model.saver.restore(sess, checkpoint_file)

operations = sess.graph.get_operations()
for op in operations:
    for tensor in op.values():
        print(tensor)

output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output/predictions'])
with tf.gfile.FastGFile('pb_model/model.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())
import tensorflow as tf

from config import Config

max_len = Config.max_sequence_length

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file="pb_model/model.pb",
                                                      input_arrays=["inputs"],
                                                      output_arrays=["output/logits"],
                                                      input_shapes={"inputs": [1, max_len]})

# converter.optimizations = ["DEFAULT"]  # 保存为v1,v2版本时使用
# converter.post_training_quantize = True  # 保存为v2版本时使用
tflite_model = converter.convert()
with open("tflite_model/model.tflite", "wb") as fw:
    fw.write(tflite_model)

### 模型训练指引

#### 1，执行python trainer.py 训练模型

#### 2，执行checkpoint_to_pb.py 将checkpoint模型转换成freeze pb模型

#### 3，执行convert_to_tflite.py 将freeze pb模型转换成tflite模型

### 注意事项
* 1，训练模型时要保证模型中不存在需要动态解释的维度，最常见的就是使用tf.placeholder传入数据时不指定数据的维度，比如batch size
    因为在转换到tflite模型，甚至之后用c++加载都需要预先确定各个tensor的维度，c++在确定tensor的维度和数据类型，就会分配一定的内存给
    这个tensor

* 2，目前tflite有很多operation不支持，比如dynamic_rnn，所以模型中换成了static_rnn，还有random_uniform，booleam_mask。
    所有在执行checkpoint_to_pb.py时指定CharRnnModel中的is_training=False，这样就可以生成一套用于预测的计算图，而这幅图里面
    去除了一些不支持的操作

* 3，实测中argmax这个operation在c++调用tflite时存在问题，所以convert_to_tflite.py中将最后的输出结果定为output/logits

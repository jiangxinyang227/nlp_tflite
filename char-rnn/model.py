
import tensorflow as tf


class CharRnnModel(object):
    """
    cbow model
    """

    def __init__(self, config, is_training=True):
        """
        init
        """
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.max_len = config.max_sequence_length
        self.num_classes = config.num_classes

        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.max_grad_norm = config.max_grad_norm
        self.learning_rate = config.learning_rate

        self.is_training = is_training

        if not self.is_training:
            self.batch_size = 1

        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_len], name="inputs")

        if self.is_training:
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.sequence_lens = tf.placeholder(tf.int32, [self.batch_size], name="sequence_lens")
            self.masks = tf.sequence_mask(self.sequence_lens, self.max_len, name="masks")
            self.keep_prob_input = tf.placeholder(tf.float32, None, name="keep_prob_input")
            self.keep_prob_lstm = tf.placeholder(tf.float32, None, name="keep_prob_lstm")
            self.keep_prob_output = tf.placeholder(tf.float32, None, name="keep_prob_output")

        self.l2_loss = tf.constant(0.0)  # 定义l2损失

        self.build_model()
        self.saver = self.init_saver()

    def build_model(self):
        """
        build model
        """

        # embedding layer
        with tf.name_scope("embedding"):
            embedding_w = tf.get_variable("embedding_w",
                                          shape=[self.vocab_size, self.embedding_size],
                                          initializer=tf.glorot_normal_initializer())

            embedding_output = tf.nn.embedding_lookup(embedding_w, self.inputs, name="embedding_words")

            if self.is_training:
                embedding_output = tf.nn.dropout(embedding_output, self.keep_prob_input)

        with tf.name_scope('lstm'):
            lstm_inputs = tf.split(embedding_output, num_or_size_splits=self.max_len, axis=1)
            lstm_inputs = [tf.reshape(lstm_input, [-1, self.embedding_size]) for lstm_input in lstm_inputs]
            for index in range(self.num_layers):
                with tf.variable_scope("lstm" + str(index)):
                    cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                    if self.is_training:
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.keep_prob_lstm)

                    initial_state = cell.zero_state(self.batch_size, tf.float32)
                    # 通过dynamic_rnn对cell展开时间维度
                    lstm_inputs, _ = tf.nn.static_rnn(cell, lstm_inputs, initial_state=initial_state)

            # 通过lstm_outputs得到概率
            seq_output = [tf.reshape(lstm_input, [-1, 1, self.hidden_size]) for lstm_input in lstm_inputs]
            seq_output = tf.concat(seq_output, axis=1)
            if self.is_training:
                mask_seq_output = tf.boolean_mask(seq_output, self.masks, name="mask_seq_output")
            else:
                mask_seq_output = tf.reshape(seq_output, [-1, self.hidden_size], name="mask_seq_output")
        # output layer
        with tf.name_scope("output"):
            output_w = tf.get_variable("output_w",
                                       shape=[self.hidden_size, self.num_classes],
                                       initializer=tf.glorot_uniform_initializer())

            output_b = tf.get_variable("output_b",
                                       shape=[self.num_classes],
                                       initializer=tf.zeros_initializer())

            self.logits = tf.nn.xw_plus_b(mask_seq_output, output_w, output_b, name="logits")
            self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        if self.is_training:
            # cal loss
            with tf.name_scope("loss"):
                self.cost, self.loss = self.cal_loss()

            with tf.name_scope("train_op"):
                self.train_op, _ = self.get_train_op(self.cost)

    def init_saver(self):
        """
        init saver
        """
        variables = tf.global_variables()
        saver = tf.train.Saver(variables)
        return saver

    def cal_loss(self):
        """
        计算损失值
        """

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

        cost = tf.reduce_sum(losses) / self.batch_size
        loss = tf.reduce_mean(losses, name="loss")

        return cost, loss

    def get_train_op(self, loss):
        """
        获取训练入口operation
        args:
            loss: 损失值
        """

        # 创建优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # 计算梯度值
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(loss, trainable_params)

        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        # 保存损失值到tensorboard中
        tf.summary.scalar("loss", loss)
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def train(self, sess, batch, config):
        """
        标准的softmax train入口
        args:
            sess: tf.Session()对象
            batch: batch size 的训练数据
            config: 配置参数对象
        """

        feed_dict = {
            self.inputs: batch["inputs"],
            self.labels: batch["labels"],
            self.sequence_lens: batch["sequence_lens"],
            self.keep_prob_input: config.keep_prob_input,
            self.keep_prob_lstm: config.keep_prob_lstm,
            self.keep_prob_output: config.keep_prob_output
        }

        _, loss = sess.run([self.train_op, self.loss],
                           feed_dict=feed_dict)

        return loss

    def eval(self, sess, batch):
        """
        标准的softmax train入口
        args:
            sess: tf.Session()对象
            batch: batch size 的训练数据
        """
        feed_dict = {
            self.inputs: batch["inputs"],
            self.labels: batch["labels"],
            self.sequence_lens: batch["sequence_lens"],
            self.keep_prob_input: 1.0,
            self.keep_prob_lstm: 1.0,
            self.keep_prob_output: 1.0
        }

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

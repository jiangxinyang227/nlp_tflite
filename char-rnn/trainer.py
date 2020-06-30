
import os
import time

import tensorflow as tf

from metrics import cal_perplexity, mean
from config import Config
from data_helper import DataHelper
from model import CharRnnModel
from log import get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer(object):
    """
    trainer
    """

    def __init__(self):
        """
        init
        """
        self.config = Config
        self.logger = get_logger("log.txt")

        self.train_data, self.eval_data = self.load_data()
        self.model = self.create_model()

    def load_data(self):
        """
        load train and eval data
        """

        train_data, eval_data = DataHelper(self.config.train_data_path,
                                           self.config.eval_data_path,
                                           self.config.output_path,
                                           self.config.max_sequence_length).gen_data()
        return train_data, eval_data

    def create_model(self):
        """
        create model
        """
        model = CharRnnModel(self.config)
        return model

    def train(self):
        """
        train
        """
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    per_process_gpu_memory_fraction=1.0)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
        )
        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())

            current_step = 0

            for epoch in range(self.config.epochs):
                start = time.time()
                print("-------- Epoch {}/{} -------".format(epoch + 1, self.config.epochs))

                for batch in DataHelper.next_batch(self.train_data, self.config.batch_size):
                    current_step += 1

                    loss = self.model.train(sess, batch, self.config)
                    perplexity = cal_perplexity(loss)

                    if current_step % 100 == 0:
                        print("train ---> step: {}, loss: {}, perplexity: {}".format(current_step, loss, perplexity))

                    if current_step % self.config.eval_every == 0:

                        eval_losses = []
                        eval_perplexities = []

                        for eval_batch in DataHelper.next_batch(self.eval_data, self.config.batch_size):
                            eval_loss = self.model.eval(sess, eval_batch)
                            eval_perplexity = cal_perplexity(eval_loss)
                            eval_losses.append(eval_loss)
                            eval_perplexities.append(eval_perplexity)

                        print("\n")
                        print("eval ---> step: {}, loss: {}, perplexity: {}".format(current_step,
                                                                                    mean(eval_losses),
                                                                                    mean(eval_perplexities)))
                        print("\n")

                        # 保存checkpoint model
                        ckpt_model_path = self.config.ckpt_model_path
                        if not os.path.exists(ckpt_model_path):
                            os.makedirs(ckpt_model_path)
                        ckpt_model_path = os.path.join(ckpt_model_path, "model")
                        self.model.saver.save(sess, ckpt_model_path, global_step=current_step)

                end = time.time()
                print("------time: {}----------".format(end - start))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

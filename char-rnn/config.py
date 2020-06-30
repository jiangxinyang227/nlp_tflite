
class Config(object):
    """
    配置模型的参数
    """

    epochs = 100
    checkpoint_every = 1000
    eval_every = 1000
    
    learning_rate = 1.0
    max_grad_norm = 3

    batch_size = 32
    
    vocab_size = 10001
    num_classes = 10000
    max_sequence_length = 70

    embedding_size = 400
    hidden_size = 400
    num_layers = 3

    keep_prob_input = 0.6
    keep_prob_lstm = 0.75
    keep_prob_output = 0.6

    train_data_path = "data/ptb.train.txt"
    eval_data_path = "data/ptb.valid.txt"
    output_path = "output"
    ckpt_model_path = "ckpt_model"

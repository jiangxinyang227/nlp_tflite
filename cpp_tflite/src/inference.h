//
// Created by Jiang,Xinyang on 2020/6/22.
//

#ifndef INFERENCE_TFLITE_INFERENCE_H
#define INFERENCE_TFLITE_INFERENCE_H
# include <string>

namespace stroke_model {

    struct Settings {
        bool verbose = true;
        std::string model_file = "/Users/jiangxinyang/Desktop/nlp_tflite/char-rnn/tflite_model/model.tflite";
        std::string word_to_index_file = "/Users/jiangxinyang/Desktop/nlp_tflite/char-rnn/output/word_to_index.txt";
        std::string label_to_index_file = "/Users/jiangxinyang/Desktop/nlp_tflite/char-rnn/output/label_to_index.txt";
        int number_of_threads = 1;
    };

}


#endif //INFERENCE_TFLITE_INFERENCE_H

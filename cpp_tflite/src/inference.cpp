
#include <sys/uio.h>    // NOLINT(build/include_order)

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "inference.h"
#include "../include/tensorflow/lite/kernels/register.h"
#include "../include/tensorflow/lite/model.h"
#include "../include/tensorflow/lite/optional_debug_tools.h"
#include "../include/tensorflow/lite/string_util.h"

#define LOG(x) std::cerr

namespace stroke_model {

    class StrokeModel {
    public:
        StrokeModel(Settings *s);

        ~StrokeModel() {};

        std::vector<std::string> run_inference(std::vector<std::string> &inputs,
                                               int sequence_len);

    private:
        void read_word_to_index(const std::string &word_to_index_file);

        void read_index_to_label(const std::string &label_to_index_file);

        // model一定也要作为成员属性，model这个指针需要一些存在
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> cur_interpreter; // tflite 解释器

        // 模型推断时依赖的参数
        std::map<std::string, int> word_to_index;
        std::map<int, std::string> index_to_label;

        const int max_length = 70;
        const int num_classes = 10000;
    };

/*
@param word_to_index_file：字母映射到索引的文件
@desc 读取字母到索引的映射表，写入到map中
*/
    void StrokeModel::read_word_to_index(const std::string &word_to_index_file) {
        std::ifstream infile;
        infile.open(word_to_index_file.data());
        assert(infile.is_open());

        std::string temp;
        while (getline(infile, temp)) {
            std::string::size_type pos = temp.find("\t");
            std::string word = temp.substr(0, pos);
            std::string index = temp.substr(pos + 1);
            int int_index = atoi(index.c_str());

            word_to_index.insert(std::pair<std::string, int>(word, int_index));
        }
        infile.close();
    }

/*
@param label_to_index_file：标签映射到索引的文件
@desc 读取字母到索引的映射表，写入到map中
*/
    void StrokeModel::read_index_to_label(const std::string &label_to_index_file) {
        std::ifstream infile;
        infile.open(label_to_index_file.data());
        assert(infile.is_open());

        std::string temp;
        while (getline(infile, temp)) {
            std::string::size_type pos = temp.find("\t");
            std::string word = temp.substr(0, pos);
            std::string index = temp.substr(pos + 1);
            int int_index = atoi(index.c_str());
            index_to_label.insert(std::pair<int, std::string>(int_index, word));
        }
        infile.close();
    }

/*
@param s: 模型推断依赖参数
@desc: 初始化时加载解释器和词索引表
*/
    StrokeModel::StrokeModel(Settings *s) {
        // 读取词索引表
        read_word_to_index(s->word_to_index_file);
        read_index_to_label(s->label_to_index_file);

        if (!s->model_file.c_str()) {
            LOG(ERROR) << "no model file name\n";
            exit(-1);
        }

        // 1，创建模型和解释器对象，并加载模型

        model = tflite::FlatBufferModel::BuildFromFile(s->model_file.c_str());
        if (!model) {
            LOG(FATAL) << "\nFailed to mmap model " << s->model_file << "\n";
            exit(-1);
        }
        LOG(INFO) << "Loaded model " << s->model_file << "\n";
        model->error_reporter();
        LOG(INFO) << "resolved reporter\n";

        // 2，将模型中的tensor映射写入到解释器对象中
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&cur_interpreter);

        if (!cur_interpreter) {
            LOG(FATAL) << "Failed to construct interpreter\n";
            exit(-1);
        }

        // interpreter->UseNNAPI(s->accel);
        // interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

        if (s->verbose) {
            LOG(INFO) << "tensors size: " << cur_interpreter->tensors_size() << "\n";
            LOG(INFO) << "nodes size: " << cur_interpreter->nodes_size() << "\n";
            LOG(INFO) << "inputs: " << cur_interpreter->inputs().size() << "\n";
            LOG(INFO) << "input(0) name: " << cur_interpreter->GetInputName(0) << "\n";

            int t_size = cur_interpreter->tensors_size();
            for (int i = 0; i < t_size; i++) {
                if (cur_interpreter->tensor(i)->name)
                    LOG(INFO) << i << ": " << cur_interpreter->tensor(i)->name << ", "
                              << cur_interpreter->tensor(i)->bytes << ", "
                              << cur_interpreter->tensor(i)->type << ", "
                              << cur_interpreter->tensor(i)->params.scale << ", "
                              << cur_interpreter->tensor(i)->params.zero_point << "\n";
            }
        }

        if (s->number_of_threads != -1) {
            cur_interpreter->SetNumThreads(s->number_of_threads);
        }

        if (cur_interpreter->AllocateTensors() != kTfLiteOk) {
            LOG(FATAL) << "Failed to allocate tensors!";
        }

//    if (s->verbose) PrintInterpreterState(cur_interpreter.get());
    }

    std::vector<std::string> StrokeModel::run_inference(std::vector<std::string> &inputs,
                                                        int sequence_len) {

        std::vector<std::string> outputs;

        // 3，定义输入和输出tensor对象
        const std::vector<int> input_tensors = cur_interpreter->inputs();
        int32_t *input_data = cur_interpreter->typed_tensor<int32_t>(input_tensors[0]);

        // 1，将输入的单词转换成索引，并将数据写入到输入的tensor中
        std::cout << "input data: ";
        for (int i = 0; i < max_length; i++) {
            std::map<std::string, int>::iterator iter;
            std::cout << inputs[i] << " ";
            iter = word_to_index.find(inputs[i]);
            int32_t index = (int32_t) iter->second;
            input_data[i] = index;
        }
        std::cout << std::endl;
        std::cout << "data write finished!" << std::endl;

        TfLiteStatus status = cur_interpreter->Invoke();

        // 2，进行推断
        if (cur_interpreter->Invoke() != kTfLiteOk) {
            std::cout << "invoke failure" << std::endl;
        } else {
            std::cout << "invoke succesed" << std::endl;
        }

        // 3，获取结果
        TfLiteTensor *predict_tensor = cur_interpreter->tensor(cur_interpreter->outputs()[0]);
        float *predict_data = predict_tensor->data.f;

        if (!predict_data)
            LOG(INFO) << "null pointer!\n";
        else{
            for (int i = 0; i < sequence_len; i ++){
                std::vector<float> class_probs;
                for (int j = 0; j < num_classes; j++){
                    class_probs.push_back(predict_data[j]);
                }
                predict_data += num_classes;

                std::vector<float>::iterator max_val = std::max_element(std::begin(class_probs), std::end(class_probs));
                int max_index = std::distance(std::begin(class_probs), max_val);
                std::map<int, std::string>::iterator iter = index_to_label.find(max_index);
                std::string label = iter->second;
                outputs.push_back(label);
            }
        }
        return outputs;
    }

}  // namespace stroke_model

int main() {

    stroke_model::Settings s;

    stroke_model::StrokeModel strokeModel(&s);

    std::vector<std::string> inputs = {"what", "are", "you", "doing"};
    for (int i = 0; i < 66; i++){
        inputs.push_back("<pad>");
    }

    int sequence_len = 4;

    std::vector<std::string> outputs = strokeModel.run_inference(inputs, sequence_len);
    std::cout << "outputs: ";
    for (int i = 0; i < outputs.size(); i++){
        std::cout << outputs[i] << " ";
    }
    std::cout << "\n";
}
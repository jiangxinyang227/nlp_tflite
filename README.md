## c++调用tflite

### 一，进入到char-rnn下面训练模型，并将模型转换成tflite。详见char-rnn/README.md

### 二，编译libtensorflowlite.so文件（mac下测试，linux类似）
* 1，安装bazel。见官方教程 https://docs.bazel.build/versions/master/install.html
* 2，git clone https://github.com/tensorflow/tensorflow.git
* 3，进入到tensorflow文件夹下，也就是位于WORKSPACE下面，bazel需要在这个文件的目录下执行编译
* 4，./configure
* 5，bazel build --cxxopt='--std=c++11' //tensorflow/lite:libtensorflowlite.so  如果要在安卓上使用，需要指定ndk

### 三，进入到cpp_tflite中，引入头文件和so文件
* 1，将tflite的头文件放置到include下(cd tensorflow/tensorflow) (find ./lite -name "*.h" | tar -cf headers.tar -T -)
* 2，将上面的headders.tar压缩包移动到自己项目下的cpp_tflite/include中并解压，解压后可以直接去除
* 3，将tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so和tensorflow/bazel-bin/external/flatbuffers/libflatbuffers.a
    放到cpp_tflite/lib文件夹中
* 4，进入到tensorflow/tensorflow/lite/tools/make下执行download_dependencies.sh，或者直接下载
    flatbuffers压缩包，然后取出include的内容放置到cpp_tflite/include中即可。
* 5，编写CMakeLists.txt，链接到so文件
* 6，mkdir build
* 7，cd build
* 8，cmake ..
* 9，make
* 10，./nlp_tflite

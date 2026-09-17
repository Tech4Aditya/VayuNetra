// infer.cpp
//
// Optional real-time inference layer: loads the classifier/predictor after
// export to ONNX and runs inference in C++ via ONNX Runtime, for the live
// demo path where you want to show a compiled, optimized inference engine
// instead of a Python process. This is the legitimate place for C++ in an
// ML pipeline — NOT for training the models themselves.
//
// Build (after installing onnxruntime C++ package):
//   g++ infer.cpp -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib \
//       -lonnxruntime -o cyclone_infer
//
// Export models to ONNX first (from Python):
//   torch.onnx.export(model, dummy_input, "classifier.onnx")
//   torch.onnx.export(model, dummy_input, "predictor.onnx")
//
// Usage:
//   ./cyclone_infer classifier.onnx frame.bin 64
//   ./cyclone_infer predictor.onnx sequence.bin 64 8

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <numeric>

std::vector<float> read_raw_floats(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open input file: " + path);
    }
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!f) {
        throw std::runtime_error("File shorter than expected tensor size: " + path);
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <raw_float32_input.bin> <size> [seq_len]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    int size = std::stoi(argv[3]);
    int seq_len = (argc >= 5) ? std::stoi(argv[4]) : 1;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cyclone_infer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);

    size_t total_elems = static_cast<size_t>(seq_len) * size * size;
    std::vector<float> input_data = read_raw_floats(input_path, total_elems);

    std::vector<int64_t> input_shape = (seq_len > 1)
        ? std::vector<int64_t>{1, seq_len, 1, size, size}
        : std::vector<int64_t>{1, 1, size, size};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* out_data = output_tensors.front().GetTensorMutableData<float>();
    auto out_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    size_t out_count = 1;
    for (auto d : out_shape) out_count *= static_cast<size_t>(d);

    std::cout << "Output (" << out_count << " values): ";
    for (size_t i = 0; i < out_count; ++i) std::cout << out_data[i] << " ";
    std::cout << std::endl;

    return 0;
}

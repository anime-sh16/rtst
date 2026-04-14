#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace torch::executor;

// ImageNet standardization values
const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

// I/O helpers

/// Read raw float32 tensor from binary file (C-contiguous, no header).
std::vector<float> read_raw_tensor(const std::string& path, size_t num_elements) {

    std::ifstream inFile(path, std::ios::binary);

    if (!inFile.is_open()) {
        throw std::runtime_error("Could not open the input tensor-binary file");
    }

    std::streamsize fileSize = static_cast<std::streamsize>(num_elements * sizeof(float));
    std::vector<float> tensorData(num_elements);


    if (!inFile.read(reinterpret_cast<char*>(tensorData.data()), fileSize)) {
        throw std::runtime_error("Error occurred while reading the file");
    }

    return tensorData;
}

/// Write raw float32 tensor to binary file (C-contiguous, no header).
bool write_raw_tensor(
    const float* data, size_t num_elements, const std::string& path) {

    std::ofstream outFile(path, std::ios::binary);

    if (!outFile.is_open()) {
        return false;
    }

    std::streamsize fileSize = static_cast<std::streamsize>(num_elements * sizeof(float));
    outFile.write(reinterpret_cast<const char*>(data), fileSize);

    bool success = outFile.good();
    outFile.close();

    return success;
}

// Image helpers (for infer mode only)

/// Load image, resize shortest edge to max(h,w), center-crop to (h,w),
/// normalize with ImageNet stats. Returns CHW float32 vector.
std::vector<float> load_image(
    const std::string& path, int target_h, int target_w) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not read image at " << path << std::endl;
        return {};
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    int max_dim = std::max(target_h, target_w);
    int new_w, new_h;
    if (img.cols < img.rows) {
        new_w = max_dim;
        new_h = (img.rows * max_dim) / img.cols;
    } else {
        new_h = max_dim;
        new_w = (img.cols * max_dim) / img.rows;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int start_x = (new_w - target_w) / 2;
    int start_y = (new_h - target_h) / 2;
    cv::Rect crop_region(start_x, start_y, target_w, target_h);
    cv::Mat cropped = resized(crop_region);

    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1.0f / 255.0f);

    std::vector<float> chw_tensor(3 * target_h * target_w);
    for (int h = 0; h < target_h; ++h) {
        for (int w = 0; w < target_w; ++w) {
            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                float normalized_val = (pixel[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                int chw_index = (c * target_h * target_w) + (h * target_w) + w;
                chw_tensor[chw_index] = normalized_val;
            }
        }
    }
    return chw_tensor;
}

/// Save CHW float32 tensor (values in [0,1]) to image file.
/// Clamps to [0,1], converts to BGR 8-bit, writes via OpenCV.
void save_image(
    const float* data, int target_h, int target_w, const std::string& path) {
    cv::Mat out_img(target_h, target_w, CV_32FC3);

    for (int h = 0; h < target_h; ++h) {
        for (int w = 0; w < target_w; ++w) {
            cv::Vec3f pixel;
            for (int c = 0; c < 3; ++c) {
                int chw_index = (c * target_h * target_w) + (h * target_w) + w;
                // Clamp value to [0, 1] as expected by your python logic
                float val = std::max(0.0f, std::min(1.0f, data[chw_index]));
                // OpenCV expects BGR output
                pixel[2 - c] = val;
            }
            out_img.at<cv::Vec3f>(h, w) = pixel;
        }
    }

    cv::Mat final_img;
    out_img.convertTo(final_img, CV_8UC3, 255.0f);
    cv::imwrite(path, final_img);
    std::cout << "Image saved to: " << path << std::endl;
}

// ExecuTorch forward pass (shared by both modes)

/// Load .pte model and run forward on the given input buffer.
/// On success, returns the output tensor data as a vector.
/// On failure, returns an empty vector.
std::vector<float> run_forward(
    const std::string& model_path,
    float* input_data,
    int32_t sizes[],  // e.g. {1, 3, H, W}
    int ndim) {

    Module module(model_path);

    TensorImpl impl(ScalarType::Float, ndim, sizes, input_data);
    Tensor input_tensor(&impl);

    std::vector<EValue> inputs = {EValue(input_tensor)};
    auto result = module.forward(inputs);

    if (result.ok()) {
        std::cout << "Module forward pass Successful!" << std::endl;

        auto outputs = result.get();

        std::vector<float> output_data(outputs[0].toTensor().data_ptr<float>(), outputs[0].toTensor().data_ptr<float>() + outputs[0].toTensor().numel());
        return output_data;

    } else {
        std::cerr << "❌ Module forward pass failed with error code: " << (int)result.error() << std::endl;

        return std::vector<float>();
    }

}

// Mode: infer  (image in → image out)
// Usage: ./rtst infer <model.pte> <input_image> <output_image> <H> <W>

int cmd_infer(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " infer <model.pte> <input.jpg> <output.jpg> <H> <W>\n";
        return 1;
    }

    std::string model_path = argv[2];
    std::string input_image = argv[3];
    std::string output_image = argv[4];
    int h = std::stoi(argv[5]);
    int w = std::stoi(argv[6]);

    std::vector<float> image_data = load_image(input_image, h, w);

    if (image_data.empty()) {
        std::cerr << "❌ Failed to load image " << input_image << std::endl;
        return 1;
    }

    int32_t sizes[] = {1, 3, h, w};
    std::vector<float> output_data = run_forward(model_path, image_data.data(), sizes, 4);

    if (output_data.empty()) {
        std::cerr << "❌ Failed to run forward pass" << std::endl;
        return 1;
    }

    save_image(output_data.data(), h, w, output_image);

    std::cout << "Output image saved!" << std::endl;
    return 0;
}

// Mode: validate  (raw tensor in → raw tensor out, no image processing)
// Usage: ./rtst validate <model.pte> <input.bin> <output.bin> <H> <W>

int cmd_validate(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " validate <model.pte> <input.bin> <output.bin> <H> <W>\n";
        return 1;
    }

    std::string model_path = argv[2];
    std::string input_bin = argv[3];
    std::string output_bin = argv[4];
    int h = std::stoi(argv[5]);
    int w = std::stoi(argv[6]);

    size_t num_elements = 1 * 3 * h * w;

    std::vector<float> input_data = read_raw_tensor(input_bin, num_elements);

    if (input_data.empty()) {
        std::cerr << "❌ Failed to load input tensor " << input_bin << std::endl;
        return 1;
    }

    int32_t sizes[] = {1, 3, h, w};
    std::vector<float> output_data = run_forward(model_path, input_data.data(), sizes, 4);
    float *output_data_ptr = output_data.data();

    if (output_data.empty()) {
        std::cerr << "❌ Failed to run forward pass" << std::endl;
        return 1;
    }

    bool write_status =write_raw_tensor(output_data_ptr, num_elements, output_bin);

    if (write_status){
        std::cout << "Output tensor saved!" << std::endl;
        return 0;
    }
    else {
        std::cerr << "❌ Failed to save output tensor" << std::endl;
        return 1;
    }
}

// Main dispatcher

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <infer|validate> ...\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "infer") {
        return cmd_infer(argc, argv);
    } else if (mode == "validate") {
        return cmd_validate(argc, argv);
    } else {
        std::cerr << "Unknown mode: " << mode
                  << ". Use 'infer' or 'validate'.\n";
        return 1;
    }
}

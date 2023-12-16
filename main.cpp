#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yolov5Inference.h"
#define DEVICE 0  // GPU id



// stuff we know about the network and the input/output blobs
// static const int INPUT_H = Yolo::INPUT_H;
// static const int INPUT_W = Yolo::INPUT_W;
// static const int CLASS_NUM = Yolo::CLASS_NUM;
// static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

//Jie Addition
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir,float& conf_th) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net == "s") {
            gd = 0.33;
            gw = 0.50;
        } else if (net == "m") {
            gd = 0.67;
            gw = 0.75;
        } else if (net == "l") {
            gd = 1.0;
            gw = 1.0;
        } else if (net == "x") {
            gd = 1.33;
            gw = 1.25;
        } else if (net == "c" && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        std::string conf = std::string(argv[4]);
        conf_th = std::atof(conf.c_str());
    } else {
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    //这里需要注意这个参数输入是否与yolov5一致
    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    float conf_th = 0.0f;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir,conf_th)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s [.wts] [.engine] [s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d [.engine] [sample_file] [conf_th]  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // 这两句后面YOLOv5Infer有定义
    // char *trtModelStream{ nullptr }; 
    // size_t size{ 0 };

    // std::string engine_name = STR2(NET);
    // engine_name = "yolov5" + engine_name + ".engine";

    // 输入参数只有两个的时候才执行？
    // if (argc == 2 && std::string(argv[1]) == "-s") { 
    if (!wts_name.empty()) {
        return create_enginefile(BATCH_SIZE,gd,gw,wts_name,engine_name);
    } 

    YOLOv5Infer yolov5Detector = YOLOv5Infer(engine_name);

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    //读取文件夹图片来推理
    for (int f = 0; f < (int)file_names.size(); f++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
            std::vector<cv::Rect> boxes;
            std::vector<float> confs;
            std::vector<float> cls_id;
            yolov5Detector.run(img,boxes,confs,cls_id,conf_th);
            cv::imwrite(img_dir+"/_" + file_names[f], img);
        }

    return 0;
}

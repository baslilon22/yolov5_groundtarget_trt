#ifndef YOLOV5INFERENCE_H_
#define YOLOV5INFERENCE_H_

#include "yololayer.h"
#define BATCH_SIZE 1
#include "preprocess.h"
#include <iostream>
#include <chrono>
#include <cmath>
// #include "cuda_utils.h"
#include "logging.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"


// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no 

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
bool create_enginefile(unsigned int batch_size,float gd,float gw,std::string wts_name,std::string engine_name);

class YOLOv5Infer
{
public:
    void* buffers[2];

    int inputIndex;
    int outputIndex;

    YOLOv5Infer(std::string engine_name);

    void run(cv::Mat& img,std::vector<cv::Rect>& boxes,std::vector<float> &confs,std::vector<float> &cls_id,float conf_th);

    ~YOLOv5Infer();

};

#endif
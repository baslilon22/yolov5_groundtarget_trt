#include "yolov5Inference.h"
#include "common.hpp"
#include "utils.h"
#include "preprocess.h"
// #include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define NMS_THRESH 0.1
#define CONF_THRESH 0.1
#define STR1(x) #x
#define STR2(x) STR1(x)
#define DEVICE 0  // GPU id

IRuntime* runtime;
IExecutionContext* context;
cudaStream_t stream;
ICudaEngine* engine;



const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
static float prob[BATCH_SIZE * OUTPUT_SIZE];


static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 185), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 185), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 185), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    // std::cout<<"yolo output shape:"<<yolo->getOutput(0)->getDimensions().nbDims<<std::endl;
    for (int i=0;i<yolo->getOutput(0)->getDimensions().nbDims;i++){
        std::cout<<yolo->getOutput(0)->getDimensions().d[i]<<std::endl;
    }
    // std::cout<<"det0 output shape:"<<det0->getOutput(0)->getDimensions().nbDims<<std::endl;
    for (int i=0;i<det0->getOutput(0)->getDimensions().nbDims;i++){
        std::cout<<det0->getOutput(0)->getDimensions().d[i]<<std::endl;
    }
    // std::cout<<"det1 output shape:"<<det1->getOutput(0)->getDimensions().nbDims<<std::endl;
    for (int i=0;i<det1->getOutput(0)->getDimensions().nbDims;i++){
        std::cout<<det1->getOutput(0)->getDimensions().d[i]<<std::endl;
    }
    // std::cout<<"det2 output shape:"<<det2->getOutput(0)->getDimensions().nbDims<<std::endl;
    for (int i=0;i<det2->getOutput(0)->getDimensions().nbDims;i++){
        std::cout<<det2->getOutput(0)->getDimensions().d[i]<<std::endl;
    }
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));

    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine_s(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}


bool create_enginefile(unsigned int batch_size,float gd,float gw,std::string wts_name,std::string engine_name){
    IHostMemory* modelStream{ nullptr };
    APIToModel(batch_size, &modelStream, wts_name);
    assert(modelStream != nullptr);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}



void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


YOLOv5Infer::YOLOv5Infer(std::string engine_name)
{
    //载入engine
    cudaSetDevice(DEVICE);
    // deserialize the .engine and run inference
    std::cout<<"loading engine:"<<engine_name<<std::endl;
    std::ifstream file(engine_name, std::ios::binary);
    char *trtModelStream = nullptr;
    size_t size = 0;
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else{
        std::cout<<"fail to load engine:"<<engine_name<<std::endl;
    }


    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::cout<<"YOLOv5 init finished!"<<std::endl;
}


YOLOv5Infer::~YOLOv5Infer()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void YOLOv5Infer::run(cv::Mat& img,std::vector<cv::Rect>& boxes,std::vector<float> &confs,std::vector<float> &cls_id,float conf_th)
{

    int fcount = BATCH_SIZE;
    auto before = std::chrono::system_clock::now();
    // if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
    for (int b = 0; b < fcount; b++) {
        if (img.empty()) continue;
        cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
        int step = pr_img.step;
        uchar* image_data = pr_img.data;
        ToChannelLast_GPU(data,image_data,step,INPUT_H, INPUT_W,b,fcount);
    }
    auto preend = std::chrono::system_clock::now(); 
    std::cout <<"   (RotatedBox) preprocess time:" << std::chrono::duration_cast<std::chrono::milliseconds>(preend - before).count() << "ms" << std::endl;



    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout <<"   (RotatedBox) inference time:"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
    
    auto rotatednms_start = std::chrono::system_clock::now();
    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        // printf("conf_th = %f",conf_th);
        rotate_nms(res, &prob[b * OUTPUT_SIZE], conf_th, NMS_THRESH);
    }
    auto rotatednms_end = std::chrono::system_clock::now();
    std::cout <<"   (RotatedBox) rotate_nms time:"<< std::chrono::duration_cast<std::chrono::milliseconds>(rotatednms_end - rotatednms_start).count() << "ms" << std::endl;
    
    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];

        //cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
        for (size_t j = 0; j < res.size(); j++) {
                
            std::vector<cv::Point> r = plot_one_rotated_box(img,res[j].bbox,res[j].theta-179.9); 
            // string s =  "[xyls,theta] : "+r.x+" "+r.y+" "+r.width+" "<<r.height<<" "<<res[j].theta<<" cls_id : "<<res[j].class_id;
            // std::cout<<"res: [xyls,theta] : "<<r.x<<" "<<r.y<<" "<<r.width<<" "<<r.height<<" "<<res[j].theta<<" cls_id : "<<res[j].class_id<<std::endl;
            // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            
            cv::putText(img, std::to_string((int)res[j].class_id)+" "+std::to_string((int)res[j].theta), r[0], cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 0), 2);
            //这里有改动
            // if(res[j].conf < conf_th)continue;
            // cv::Rect r = get_rect(img, res[j].bbox);
            // r.x = std::max(r.x,0);
            // r.y = std::max(r.y,0);
            // r.width = std::min(img.cols-r.x,r.width);
            // r.height = std::min(img.rows-r.y,r.height);
            // boxes.push_back(r);
            // confs.push_back(res[j].conf);
            // cls_id.push_back(res[j].class_id);
        }
    }
}




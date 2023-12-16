# yolov5_GroundTarget_trt

对地目标检测算法的tensorrt加速版，在正常框的tensorrt转换代码上。修改了yololayer层的解析代码和旋转框极大值抑制rotatedBox_nms代码

## Usage

```
mkdir build
cd build
cmake ..
make -j8

```
- 如果还没转换好engine文件，执行以下命令先转换

`./yolov5 -s`

- 转好engine文件后，执行以下命令运行

`./yolov5 -d yolov5s.engine ../samples/`
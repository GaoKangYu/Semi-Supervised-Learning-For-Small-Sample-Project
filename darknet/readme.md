# Yolo配置
- 配置darknet环境即可，本项目基于yolo v3完成验证实验
- 项目地址为：https://github.com/AlexeyAB/darknet
- 常用指令：
```
1、训练模型并映射mAP与loss曲线到 服务器ip:8090 ./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map
2、验证模型mAP ./darknet detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights
3、提取模型前81层backbone ./darknet partial cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.81 81
```

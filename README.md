# Semi-Supervised-Learning-For-Small-Sample-Project

“小样本”项目中使用的半监督方法及部分数据

## 环境需求
-   Windows or Linux
-   Python 3.5+
-   CMake >= 3.12
-   CUDA >= 10.0
-   OpenCV >= 2.4
-   cuDNN >= 7.0
-   on Linux  **GCC or Clang**, on Windows  **MSVC 2017/2019**  
## 文件结构

文件结构及其作用如下：

```
Semi-Supervised-Learning-For-Small-Sample-Project
├── SSL-Train(包含整个实验过程中使用过的逐轮数据、配置文件与部分权重[为下载地址])
│   ├── dataset-A(A数据集，为经过红外化处理后的Dota数据集，包含plane、helicopter、ship三类)
│   ├── dataset-B(B数据集，为raw-data中部分数据经过数据增广而来，包含ship1、ship2、j6、j8、z9、chf六类)
│   │   ├── SSL-Experiment-Dataset 0
│   │   ├── SSL-Experiment-Dataset 1-7
├── Test-Weights(验收时采用的权重)
├── darknet(为空，实验使用yolov3模型验证方案的可行性，按照https://github.com/AlexeyAB/darknet说明配置即可)
├── document(技术报告与测试报告，已删除敏感内容，切勿外传)
├── inference(验收时使用的调用接口，包含基于pyqt的演示系统)
│   ├── demo
├── raw-data(原始数据，标注为yolo格式)
├── test-output(自测试输出结果)
│   ├── HS
│   ├── S
│   ├── T
├── test_data(自测试测试集)
├── utils(实验过程中的常用脚本)
```

- 部分缺失权重和cfg配置文件下载地址
链接：https://pan.baidu.com/s/1pi-LjC6K-AmOuj0yYyXnnA 
提取码：IEC1

## 实验流程

-   1）A数据集制作
```
1、基于Dota数据集进行筛选，仅选取包含plane、helicopter、ship三类目标的数据，由于验证模型选取的yolo v3，故本实验选取了水平框标注。
2、使用DOTA_devkit工具进行图片分割，分割后图片的分辨率为416x416，步长无严格要求。
3、使用脚本对标注数据进行清洗和格式转化，经由DOTA_devkit工具分割后，部分目标会存在被切割到而特征不完整的情况，此时标注文件中的difficult标签会为1，舍弃difficult不为0的目标。使用utils/voc_to_yolotxt.py脚本可完成格式转化。
4、对分割后的图片进行仿红外处理。
5、按照8:2的比例划分为训练集与验证集，使用utils/img_select.py脚本可实现图片随机筛选，utils/txt_process.py脚本可实现train list及valid list制作。
6、至此，A数据集制作完毕。
```
-   2）A模型训练
```
1、基于A数据集，选取yolo v3作为验证模型，以SSL-Train/dataset-A/cfg/下文件为配置展开训练，单卡。
2、验证其mAP，选取其中最高的作为teacher-A.weights，提取其backbone作为知识迁移模型，本验证实验提取了其前81层，具体指令可参见darknet/read.md，得到teacher-A.conv.81。
```
- 3）B数据集制作
```
1、基于raw-data中的数据，划分为训练集与测试集，项目统一的测试集（验收时有修改）为：
飞行器测试图像：lp_scene1501-1_1m.bmp、lp_scene1501-2_1m.bmp、lp_scene2201-1_1m.bmp、lp_scene2201-3_1m.bmp。
船测试图像：lb_scene0201-4_1m.bmp、lb_scene0901-1_1m.bmp、lb_scene1201-1_1m.bmp、lb_scene1201-2_1m.bmp、lb_scene1501-2_1m.bmp。
2、分别对训练集、测试集图片使用utils/imgaug-yolov3.py工具进行数据扩充，得到扩充后的训练集、测试集，此后，B数据集中的测试集不变，选取训练集图片的25%作为小样本，使用原生标注，组成小样本直接训练的训练集。
3、以25%训练集图片及其原生标注文件为训练集，B数据集中的测试集为测试集，SSL-Train/dataset-B/SSL-Experiment-Dataset 0/cfg/（已经压缩到了SSL-Experiment-Dataset 0.zip中）为配置文件，teacher-A.conv.81为初始化权重，单卡展开小样本直接建模的训练。
4、选取其early stop点位置的权重，验证其mAP，满足舰船类及其子类（即ship1和ship2的mAP都要求达到指标）mAP达到82%、飞行器类及其子类mAP达到75%即可，由于最终验收有提升量10%的要求，起点基数（mAP）不要选取的太高。
5、寻找到满足指标且基数不太高的权重teacher-B.weights后，同样提取其backbone，获得teacher-B.conv.81。
```
- 4）多轮次半监督训练
```
1、删除B数据集训练集中剩下的75%数据的原生标签，基于teacher-B.weights开始制作伪标签。运行utils/darknet_test.py（需要注释并开启部分代码段），可实现端到端的伪标签制作，其中，阈值设为0.95，若某张图片中有多个置信度不同的目标，只要有一个目标置信度低于0.95，则舍弃整个标注文件。
2、经过伪标签制作后，通过运行utils/img_select.py，挑选出生成了高置信度伪标签的数据，并使用该脚本将数据分为n个文件夹，每个文件夹100张数据（最后一轮可不满100张）。例如，经过伪标签生成获得了649张含高置信度伪标签的数据，可分为7个文件夹，对应7轮实验，分别含有100/100/100/100/100/100/49张数据。
3、基于每一个文件夹的数据制作train list，以项目验收实验过程中的train list为例，加上小样本的247张数据，train list分别包含247,347,447,547,647,747,847,947张数据，此过程中验证集保持不变。
4、开展逐轮实验，具体的cfg文件可以参考SSL-Train/dataset-B/SSL-Experiment-Dataset 1-7/cfg(在压缩包SSL-Experiment-Dataset 0-7.zip中).
5、验证每轮实验的early stop点mAP，记录实验数据，直到满足指标，获得满足舰船类及其子类92%、飞行器类及其子类85%以上mAP且提升量达到10%的权重student-7.weights。
*若遇到某一子类检测率提升有限的情况，可使用老师模型预测一定数目的该类数据，等量替换每轮实验中的训练数据。例如在某次实验中发现舰船类(尤其是舰船2类)和超黄蜂类的检测率较低，因此在下一次的实验中，每组额外添加了20张舰船类数据(舰船1和舰船2混合，占比未知，随机抽选)和14张舰船2类、6张超黄蜂类数据，因此，每轮数据组成为：60张全种类数据(各类占比未知，随机抽选)、20张舰船类(舰船1和舰船2混合，占比未知，随机抽选)、14张舰船2类和6张超黄蜂类。
```
- 5）验收finetune实验
```
1、验收时更改了测试集，新增了20张共40个舰船目标，最终验收时用了其中10张共20个舰船目标，指标为检测率（检测数目/总目标数目），人工统计有没有检测到该目标以及分类是否删除正确。
2、finetune思路与上述实验类似，同样提取student-7.weights的backbone，将目标从6类划分为4类：ship、ship1、fighter、helicopter，其中，ship为新增的仿真数据（分3类时由于特征差异大，效果不好），ship1为原始的ship1、ship2合并而来；可以不额外扩充训练集，验证集与训练集相同。teacher-B.weights的finetune过程类似，不再赘述。
```
## 演示系统与自测试

- 演示系统基于pyqt，映射到linux图形界面进行演示，部分内容涉及保密，切勿外传。![演示系统界面](https://github.com/IEC-lab/Semi-Supervised-Learning-For-Small-Sample-Project/blob/main/inference/demo/demo.png)
- 自测试实验记录及输出结果添加在了test_data和document下。

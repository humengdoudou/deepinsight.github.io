---
layout: post
category: deep_learning
title: 关于神经网络(CNN)处理特定问题的思考.结构篇
date: 2017-05-12
---

# 关于神经网络(CNN)处理特定问题的思考.结构篇


## 概念

### 为什么不能只做分类

因为分类经常是ill-posed的问题。

- 你不能保证一张图片中你所标注的所述分类物体在特定位置，比如中心出现
- 你不能保证一张图片中只出现一个物体
- 你不能保证一张图片中只出现同一类物体
- 你不能保证在数据标注过程中，人的判断就是对的


### 我们关心什么问题，怎么就能解决

- 如果你不能保证一张图片中你所标注的所述分类物体在特定位置，比如中心出现，那么我们就有了Classification + Localization的任务。
- 解决办法：也就是说，我们要把最后的AveragePooling层取消掉，用一个硬编码的框（包含框中心信息、框长宽信息），使得框里的信息特别可信地属于某个类。
- 如果你不能保证一张图片中只出现一个物体/同一类物体，那么对多个物体甚至是多类的多个物体，那么我们就有了Object Detection的任务。
- 解决办法：我们要反复的在同样一张图片上投多个硬编码的框，识别出它的中信信息、长宽信息，使得框里的信息特别可信地属于某个类。
- 如果你不想要框，想要沿着感兴趣物体的边界在原图每一个像素点上面都标出来它的所属分类，以及所属的个体，那么就有了Scene Parsing/Semantic Segmentation的任务
- 解决办法：我们要让原始像素点集合成簇，常见的方法就是单点分类，多点间用CRF约束的损失保证大块属于一类。


### 如何靠结构解决问题：元方法

- Atrous/Dilation: 在非1 * 1卷积上采用长程（带孔）的采样点进行卷积，使得感受野仅使用少数几层也能有效扩张
- MultiScale: 因为最终我们关心的是原图上的像素点级别的预测，大尺度过度卷积的结果也许要和小尺度局部卷积的结果结合
- Hard/Soft Attention: 大量论文都提到单个像素点就有分类向量，先框选一部分点预测，或者对所有点的局部进行加权来预测是Hard/Soft Attention的区别。仅有Classification标注的弱监督问题适合Soft Attention，而有Segmentation标注的数据可以Hard Attention无压力，也就是说，我们在做分类、识别、分割的预测任务时，眼睛已经有预谋的向某些区域看了
- SS/ROI Proposal: Selective Search和Regional Proposal是预测备选框，以及用NMS（非极大值抑制）避免框太多。一个很好的改进是YOLOv2，尽管它使用的还是老土的VGG16 Backbone

在此，对损失函数不做介绍。


### 用什么思想解决了问题：

为了解决这些特殊问题，我们就需要引入特殊的结构。

- FCN: 避免缩图
- HourGlass: 先Conv再Deconv, Deconv有时会留下Conv时的信息，有时二线性插值，有时学习带有分数阶Kernel的Conv的操作参数。
- Pyramid: SPP和PSP两篇文章分别提到，我们仍然要使用SIFT早就有的思想对Feature排列成金字塔进行分析
- Tree: LeCun的SIFT, 中山大学的工作，分层次的、分形的分割工作在Seg上较好
- RPN/Yolo: 投框与改进
- Mask RCNN: 把投框和分割两个任务一起学习
- 其他


### 哪些工作值得注意


#### 经典元方法1：仿射变换实现注意力、Colocalization

- Spatial Transformer Networks

#### 经典元方法2：Attention

- Residual Attention Network[]()
- Not All Pixels Are Equal: Difficulty Aware[]()
- Attention to Scale: Scale-aware Semantic Image Segmentation[]()


#### 树形分割, 先是SIFT, 后还有Image Captioning(看图说话功能)

- Learning Hierarchical Features for Scene Labeling[]()
- Deep Structured Scene Parsing by Learning with Image Descriptions[]()


#### OD系列, 最后两组OD大神合流于OD/Seg联合训练任务
- Rich Feature[]()
- Regional CNN[]()
- Fast-RCNN[]()
- Faster-RCNN[]()
- Mask-RCNN[]()


#### Seg系列重磅

- Fully Convolutional Networks for Semantic Segmentation[]()
- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation[]()
- Learning Deconvolution Network for Semantic Segmentation[]()
- Stacked Hourglass Networks for Human Pose Estimation[]()
- DeepLab



#### Seg结构最新动向

- PSPNet: Pyramid Scene Parsing Network[]()
- Wider or Deeper: []()
- FlowNet 2.0 []()
- RefineNet []()
- WildCat []()



### 如何提出问题，提出什么问题：问传感器去

- 视频拍摄：在时间轴上对单一物体的跟踪与重建的基础数据
- 多目摄像头：双眼对同一场景的同步跟踪
- 深度相机：在RGB信息上提供D的信息
- 动作捕捉：位置姿态识别
- 雷达点云：使用RADAR/LIDAR实现三维重建


### 写在最后

结构只是一个点而已，损失函数和训练技巧我们回头聊。


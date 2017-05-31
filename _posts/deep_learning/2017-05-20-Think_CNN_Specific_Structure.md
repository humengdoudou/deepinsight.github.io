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

- Spatial Transformer Networks[https://arxiv.org/pdf/1506.02025.pdf](https://arxiv.org/pdf/1506.02025.pdf)

全文重点在于提出学习仿射变换参数，能解决一大类问题。

**The transformation is then performed on the entire feature map (non-locally) and
can include scaling, cropping, rotations, as well as non-rigid deformations.**


#### 经典元方法2：Attention

- Residual Attention Network[https://arxiv.org/pdf/1704.06904.pdf](https://arxiv.org/pdf/1704.06904.pdf)
- Not All Pixels Are Equal: Difficulty Aware[https://arxiv.org/pdf/1704.01344.pdf](https://arxiv.org/pdf/1704.01344.pdf)
- Attention to Scale: Scale-aware Semantic Image Segmentation[https://arxiv.org/pdf/1511.03339.pdf](https://arxiv.org/pdf/1511.03339.pdf)


#### 树形分割套路，2017年4月中山大学VOC2012刷榜之作

继承了先人LeCun在SIFT玩儿这套, 发展了Image Captioning(看图说话)功能

- Learning Hierarchical Features for Scene Labeling[http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
- Deep Structured Scene Parsing by Learning with Image Descriptions[https://arxiv.org/pdf/1604.02271.pdf](https://arxiv.org/pdf/1604.02271.pdf)


#### OD系列关键论文

最后两组OD大神合流于OD/Seg联合训练任务。

联合训练会导致多出来的超参，这一点个人不是很喜欢，智者见智了。


- Rich Feature[https://arxiv.org/pdf/1311.2524.pdf](https://arxiv.org/pdf/1311.2524.pdf)
- Fast-RCNN[https://arxiv.org/pdf/1504.08083.pdf](https://arxiv.org/pdf/1504.08083.pdf)
- Faster-RCNN[https://arxiv.org/pdf/1506.01497.pdf](https://arxiv.org/pdf/1506.01497.pdf)
- You Only Look Once[https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640.pdf)
- Single Shot Multibox Detector[https://arxiv.org/pdf/1512.02325.pdf](https://arxiv.org/pdf/1512.02325.pdf)
- Mask-RCNN[https://arxiv.org/pdf/1703.06870.pdf](https://arxiv.org/pdf/1703.06870.pdf)


#### Seg系列重磅

我们要知道每一个像素点都属于哪个分类，传统的老板们习惯性地加CRF的方案（非端到端训练，有时成为post-hoc），这时，FCN出现了。

- Fully Convolutional Networks for Semantic Segmentation[https://arxiv.org/pdf/1411.4038.pdf](https://arxiv.org/pdf/1411.4038.pdf)

FCN里核心几句话在第三段：
- 3.1 改造：**Adapting classifiers for dense prediction**
- 3.2 不同尺度拼接： **Shift-and-stitch is filter rarefaction**
- 3.3 上采样就是非整数阶卷积，也就是带参数的二线性插值：**Upsampling is backwards strided convolution**
- 3.4 数据预处理：**Patchwise training is loss sampling**

另一类起到同样作用的方法是Dilated Conv（Atrous，带孔洞的卷积），这两者是可比较的。而且后者炫酷之处在于规避了对feature map的concat操作。至今我唯一认可的大量使用concat操作的网络结构是DenseNet.

那么根据这个Atrous我们不得不看的是Deeplab.

- DeepLab[https://arxiv.org/pdf/1606.00915.pdf](https://arxiv.org/pdf/1606.00915.pdf)

其他：

- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation[https://arxiv.org/pdf/1511.00561.pdf](https://arxiv.org/pdf/1511.00561.pdf)
- Learning Deconvolution Network for Semantic Segmentation[https://arxiv.org/pdf/1505.04366.pdf](https://arxiv.org/pdf/1505.04366.pdf)
- Stacked Hourglass Networks for Human Pose Estimation[https://arxiv.org/pdf/1603.06937.pdf](https://arxiv.org/pdf/1603.06937.pdf)


动作检测其实这个问题可以把它视作为Semantic Seg的子命题，就好像Finegrained任务是Classification任务的子命题那样。一个解决好了，另外一个自然不错。


#### Seg结构最新动向

- PSPNet: Pyramid Scene Parsing Network[https://arxiv.org/pdf/1612.01105.pdf](https://arxiv.org/pdf/1612.01105.pdf)

商汤的重磅，仍然依赖concat.

- Wider or Deeper: [https://arxiv.org/pdf/1611.10080.pdf](https://arxiv.org/pdf/1611.10080.pdf)

这篇的工作横向比较极为扎实，更像是比较Backbone Network.

- RefineNet [https://arxiv.org/pdf/1611.06612.pdf](https://arxiv.org/pdf/1611.06612.pdf)

VOC2012榜单前五的工作。


### 如何提出问题，提出什么问题：问传感器去


- 视频拍摄：在时间轴上对单一物体的跟踪与重建的基础数据
- 多目摄像头：双眼对同一场景相同角度、不同角度的同步跟踪
- 深度相机：在RGB信息上提供D的信息
- 动作捕捉：位置姿态识别
- 雷达点云：使用RADAR/LIDAR实现三维重建


### 写在最后

为各种问题提出的结构只是一个点而已，损失函数和训练技巧我们回头聊。

以及，如果你打开论文看了的话，就会对中国人做AI的水平燃起一些希望。华人很多工作很扎实，然而洋人比较喜欢鼓吹灵性、灵感一类的。因此，还是要学习一个扎实的数学和物理知识。论文里面有大量的矩阵计算，想要快速恶补一下的话，我强烈推荐：

- Matrix Cookbook[http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf)


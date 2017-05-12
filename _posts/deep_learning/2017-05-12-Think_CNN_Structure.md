---
layout: post
category: deep_learning
title: 关于卷积神经网络(CNN)骨干结构的思考
date: 2017-05-12
---

# 关于卷积神经网络(CNN)骨干结构的思考


## 概念

** 什么是机器学习、模式识别？

- 机器学习、模式识别、深度学习等等模型的目的，是压缩。对数据的背诵这不是压缩，对特征的提取才是压缩。

** 传统手工计算机视觉任务和卷积神经网络(CNN)共享哪些特性？

- 它们都抓住了、而且必须抓住平移、镜像、一定程度上的缩放不变性，只要满足不变性，相似的特征表征能力一定强。CNN的新工作还有旋转不变性、仿射变换不变性和时间轴上的灰度不变性。设计思路不包含这些不变性的，

** 一般的CNN模型，（比如在ImageNet上训练的模型）压缩效果如何？

- 在ImageNet上训练的不少模型，把2^(224 * 224 * 3 * 8bit)的数据空间中的数据特征用少到几十万，大到几亿的参数表征出来，起到了局部或者全局特征的提取，从而用特征进行各种任务。重要的是很多任务可以通过1次定义结构端到端完成。

** 用于计算机视觉分类的CNN为何重要？

- 它是Object Detection, Scene Parsing, OCR等任务的前导性任务，往往也被称为Backbone Model。而且在上面快速实验很多元方法，比如dropout，attention，套用GAN等等是直接、方便的。

** CNN存在什么问题？

- 要求概率图可微分这个假设很强，在CNN里体现为参数连续，而很多超参离散，使用同样的NN技巧自动连续超参还凑合，离散会碰到各种问题。

## 重要结构梳理

知道为什么，远比知道是什么重要。论文是读不过来的。但是读论文可以通过多读好文、浏览水文、搭配烂文提高品位。

### 我们应该关心结构的哪方面

** CNN Backbone往往是各种CNN模型的一个共享结构。

- 概念中提到，它是Object Detection, Scene Parsing, OCR等任务的前导性任务。

    - AlexNet: [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - VGG: [https://arxiv.org/pdf/1409.1556.pdf](https://arxiv.org/pdf/1409.1556.pdf)
    - Residual Network: [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
    - Wide ResNet: [https://arxiv.org/pdf/1605.07146.pdf](https://arxiv.org/pdf/1605.07146.pdf)
    - FractalNet: [https://arxiv.org/pdf/1605.07648.pdf](https://arxiv.org/pdf/1605.07648.pdf)
    - ResNeXt: [https://arxiv.org/pdf/1611.05431.pdf](https://arxiv.org/pdf/1611.05431.pdf)
    - GoogleNet: [https://arxiv.org/pdf/1409.4842.pdf](https://arxiv.org/pdf/1409.4842.pdf)
    - Inception: [https://arxiv.org/pdf/1602.07261.pdf](https://arxiv.org/pdf/1602.07261.pdf)
    - DenseNet: [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)
    - SORT: [https://arxiv.org/pdf/1703.06993.pdf](https://arxiv.org/pdf/1703.06993.pdf)
    - Compact Bilinear: [https://arxiv.org/pdf/1511.06062.pdf](https://arxiv.org/pdf/1511.06062.pdf)
    
- 这个共享结构除了结构性的超参（总深度、总宽度）以外，反复使用了多种技巧，其中包括

    - Residual(直接加)
    - Concat(特征拼接)
    - Bottleneck(特征压缩)：通过Conv(1,1)对稀疏的或者臃肿的特征进行压缩
    - Grouping(分组)：fc-softmax分类器从1个转差点把分组视为射线，分组改善了这一点
    - Fractal(分形模式)：结构复用，可能带来好处
    - High-Order(高阶)：在非分组时，可能带来好处
    - Asymmetric(非对称)
    
- 再次的，我们对结构有一个重新的审视
    
    - AlexNet/VGG: 普通
    - VGG: 加深
    - ResNet: 通过x+F(x)直接加法实现了Residual模块
    - Wide ResNet: 加宽
    - FractalNet: 结构复用，使用Concat
    - ResNeXt: ResNet基础上对Conv(3,3)使用了分组，但是如果Conv(1,1)也分组甚至精度不降
    - GoogleNet/Inception: Conv(1,3),Conv(1,5),Conv(1,7)属于非对称结构，这个技巧在OCR处理长宽非1:1的字体有用
    - DenseNet: 大量使用压缩
    - SORT: 一个小trick使用elementwise x*F(x)实现高阶
    - Compact Bilinear: 通过学习矩阵A实现x'Ay实现制造新的特征
    

** CNN Backbone以外的专用结构

- 为Object Detection设计的结构更主要关心多尺度框选准确与否，因此有RPN、SPPNet、上下采样等设计思想。
- 为Scene Parsing设计的结构更主要关心单位像素点分类正确与否，因此有FCN、上下采样等设计思想。

** 结构上的元方法

- 作为模型压缩的替代方案，XNOR，二值化（Binarization），量子化（Quantization，使用数个比特替代原模型）在本文不详细描述。


### 如何优化CNN Backbone

我们在DataParallel的语境下面讨论这个问题。

我们知道，就像组装深度学习服务器一样，你的预算一定的条件下，如何搭配一台服务器才能让CPU对数据预处理够用、内存加载数据够用、硬盘I/O够用，以及最重要的是，选择一块好的GPU卡。在这里不赘述。

进行CNN Backbone优化一样有这个问题：

** 你的显存利用率和GPU算力利用率如何达到最高？

- 降低Batch-size会减小Feature Map占用缓存，但收敛情况怎么样，可能饮鸩止渴。
- 加宽直接影响参数数量。
- 加深不仅影响参数数量还影响Feature Map大小。
- 分组极大节省参数，甚至还能提高效果。
- 结构复用、压缩节省参数，增加计算量。
- 特征拼接、高阶操作降低并行效率，尤其不是inplace的那种。
- Bilinear大量使用额外参数。
- 非对称带来额外的代码工作。
- 任何新颖结构的引入带来非连续超参，让模型BP，让超参优化无B可P。


## DeepInsight团队欢迎您来优化CNN Backbone


- 基于个人品位，使用ResNeXt+WideResNeXt+SORT。
- 大量超参可调整。
- CIFAR100 top1 acc: 84.2%，可再现。
- 计划使用Compact Bilinear和Attention。
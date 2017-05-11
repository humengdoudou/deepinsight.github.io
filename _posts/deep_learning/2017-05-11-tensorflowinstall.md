---
layout: post
category: deep_learning
title: Tensorflow on Centos7 installation notes
date: 2017-05-11
---

在Centos7上从源码安装tensorflow

(命令行先不排版)

库的预装暂不提, 我们的环境是Centos7.2 + CUDA7.5.
大体follow官方的guide: https://www.tensorflow.org/install/install_sources

首先从github clone tensorflow代码库并且checkout到r1.0版本: git checkout r1.0

执行./configure出现 "no such target '//external:android/dx_jar_import'" 错误.

git checkout 63b3fea 可解决这个问题. 但此时分支版本是0.12.(待确认1.1版本)

更新: 1.1版本没问题 - 

git branch r1.1;
git checkout r1.1;
git pull origin r1.1;

Tesla M40的capability是5.2, 不用去查了.


configure成功后执行 

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

此处坑略大, 如果你用的是自编译的gcc环境. centos7自带4.8, 我们在/usr/local下重新装了一个gcc4.9, 如果想用自编译的gcc套件, 可以参考以下网址:

https://github.com/bazelbuild/bazel/issues/649#issuecomment-166710509

http://biophysics.med.jhmi.edu/~yliu120/tensorflow.html

我尝试了一下没有解决 也因为太过繁琐放弃了. 
直接跳过这个问题, 在tensorflow configure的时候指定系统默认的/usr/bin/gcc就没问题了, 只要满足4.8+即可.
原因大致也了解了, 因为google的bazel编译工具是把gcc及其相关路径hardcode在配置里面, 比如 /usr/bin/gcc. 所以会造成不一致, 部分程序用了自带gcc, 部分用了自编的gcc.

build成功后制作pip安装包:

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

安装这个包(到用户目录):

pip install /tmp/tensorflow_pkg/tensorflow-xxx-xxx.whl --user



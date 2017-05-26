---
layout: post
category: deep_learning, tensorflow
title: Tensorflow如何用pretrained模型参数同时初始化一个网络的两个分支
date: 2017-05-25
---

How to use tensorflow pretrained model to initialize two same sub-networks under different variable scopes.

在某些情况下，我们会需要在网络中构建两个相同的分支, 这两个分支结构完全相同, 但是参数不共享. 例如在 2(4)* *Spatial Transformer Networks*. 同时需要用一个预训练好的模型参数来初始化这两个网络.



以下是示例代码: 

```python
import tensorflow as tf 
from nets.inception_v2 import inception_v2 
from nets.inception_v3 import inception_v3_arg_scope 
slim = tf.contrib.slim
ckpt_file = "./inception_v2.ckpt" 

def network_fn(inputs):
    end_points = {}
    with slim.arg_scope(inception_v3_arg_scope()): 
        end_points["end1"] = inception_v2(inputs, num_classes=200,is_training=True,scope = "inc1")  
        end_points["end2"] = inception_v2(inputs, num_classes=200,is_training=True,scope = "inc2") 
    return end_points 

data = np.zeros( (32, 224, 224, 3), dtype=np.float32) 
inputs = tf.convert_to_tensor(data)                                                                                                                                                      
end_points = network_fn(inputs)    

with tf.Session() as sess:   
    sess.run(tf.global_variables_initializer())
    #Do not restore Logits variables as they're 1000 dimensions for imagenet classification
    saver_1 = tf.train.Saver({"InceptionV2"+v.op.name[4:] : v for v in tf.global_variables() if v.name.startswith("inc1") and v.name.find("Logits/")<0})
    saver_1.restore(sess, ckpt_file)
    
    saver_2 = tf.train.Saver({"InceptionV2"+v.op.name[4:] : v for v in tf.global_variables() if v.name.startswith("inc2") and v.name.find("Logits/")<0})
    saver_2.restore(sess, ckpt_file)
```


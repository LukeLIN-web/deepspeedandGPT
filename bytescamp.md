

有现金奖励

### mentor的背景:

AML ,  高性能 , io 加速,  yi bairen 和 pengyanghua 彭杨华

Applied Machine Learning

## poster

夏令营组委会希望大家能在8月2日18:00前提交poster内容，用于8月5日游园会，组委会统一制作展板，ppt中是for营员的制作要求

提交展板制作，若项目还未产出结果，建议大家可以用【预期实现内容、预期产出结果】替代，具体展示内容不做要求（eg：项目摘要、技术原理、项目亮点等），主要是满足不同方向营员的交流与学习~"

### 队友

李想,  北邮本硕,  物流工程,  马上研二 微软亚洲研究院和腾讯实习,ACM,  报名了调度算法,  调剂到 高性能计算.  它投了其他实习流程但是被结束了. 

李想  计算所马上研二,  本科南航, 都是计算机 ,  cache 切换算法.disk上多级cache . 主要体系结构的背景.    导师是包云岗老师, 不让实习. 

笑死, 彭杨华说西瓜书我们还没看过.

### 这个项目

GPT   transformer  序列- 序列. 给定一个文本序列, 生成后面的文本或者翻译文本.   表示序列-序列的过程,  是一个NLP模型.参数非常巨大. 

好像就是 面试可以少一次面试.  

数据并行有什么不同的地方? 

https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/ 可以看这个博客里面的动画 学习

https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/ 这个是需要应用到GPT2的训练中.  应该要先复现. 一开始层太多, 可以调小层数.  

我们不使用真实数据, 下载太慢了.  fake data 测试也差不多.  GPT pretaining https://github.com/NVIDIA/Megatron-LM#gpt-pretraining

``` 
--num-layers 24  可以改成4 ,之后慢慢往上调.
text_document, vocab-file $VOCAB_FILE \
          --merge-file 这些都可以用假数据. 
          准备数据. 
```

Data Preprocessing 里面就有 教你怎么生成假数据. The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. 

尝试运行这个https://github.com/microsoft/DeepSpeedExamples/tree/25d73cf73fb3dc66faefa141b7319526555be9fc/Megatron-LM-v1.1.5-ZeRO3

先看两个blog, 然后运行这个. 

<<<<<<< HEAD
运行
=======
8.2

[Megatron-LM GPT2 - DeepSpeed](https://www.deepspeed.ai/tutorials/megatron/) 

这个已经跑通了. 

要把数据下载到 `DeepSpeedExamples/Megatron-LM/data`:

计算参数量. 

怎么说DL 有可解释性?



怎么解决问题？ 

首先是 mentor说你们可以尝试 `torch.disturbition` 和 multiple 矩阵乘法两个api  ，相乘两个 50000*50000 的矩阵， 一个矩阵10g内存，放在同一个GPU上，它显存就会爆炸。  希望把它分放到两个GPU上运行。如果放在两个GPU上它就不工作，

 多线程的方法：   一个cpu多个线程，    操控多个GPU来计算 A 矩阵划分为1/8， B 每个GPU都放一个，45秒。

 他们就尝试了一万， 十万的矩阵。



### deepspeed的加速

隐藏 ， 流水线， 计算第一层同时从cpu取第二层参数， 8月3日1.5B 十五亿可以。 阿里最顶尖的机器， 8张v100。

只到SSD , 我们是普通磁盘HDD,不支持， 

GPU 的参数缓存offload到内存。  

采用不同technology 画出几条反函数曲线。  每条 。 

先不开offload 跑

华为盘古， 阿里和清华合作， 做了上百亿参数的模型。 

一万张v100.  训练一次一千万美元。 

有哪些optimizetion， 一个个打开。

1. "gradient_accumulation_steps": 1, 

先求batch size = 4的， 得到gradient ，然后再算一个4 ， 这样相当于 batchsize  = 4。     有时候显存不够就必须把他调大。 可能和别的冲突。

2. fp16  ， 就是gpu计算时 降低精度到fp16 .  cuda 有加速fp16的硬件。 

优化方法： checkpoint， fp16，  offload

3. offload是 把中间结果 gpu上没用到的中间结果 swap到cpu上， 用到再取回来。

4. checkpoint是 中间结果占据了显存， 我们丢掉中间结果，需要用到的时候， 从第六个tensor 开始算 ，也就是从他最近的那个checkpoint 重新计算。  时间换空间。 
5. zero stage 1 2 3     ，stage =3的时候会慢， 把parameter swap 到内存上， 可以训练很大的模型 。   google 把训练好的bert给大家， 大家fine tune 一下就可以了。

>>>>>>> 30c80917670cb0edefa2db7908911ee0c32e7a07


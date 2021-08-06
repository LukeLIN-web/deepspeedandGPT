

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

运行
=======
8.2

[Megatron-LM GPT2 - DeepSpeed](https://www.deepspeed.ai/tutorials/megatron/) 

这个已经跑通了. 

要把数据下载到 `DeepSpeedExamples/Megatron-LM/data`:

计算参数量. 

怎么说DL 有可解释性?

 broadcast，  allreduce 把所有数据加起来取平均， allgather， 

怎么解决问题？ 

首先是 mentor说你们可以尝试 `torch.disturbition` 和 multiple 矩阵乘法两个api  ，相乘两个 50000*50000 的矩阵， 一个矩阵10g内存，放在同一个GPU上，它显存就会爆炸。  希望把它分放到两个GPU上运行。如果放在两个GPU上它就不工作，

 多线程的方法：   一个cpu多个线程，    操控多个GPU来计算 A 矩阵划分为1/8， B 每个GPU都放一个，45秒。 队友就尝试了一万， 十万的矩阵。



可以查询每个拓扑结构

GPU 和GPU 连接用NVlink，600GB/s 。   

CPU 和GPU  用 PCIe， 16GB/s 

CPU和CPU 用QPI

网卡 NIC 经过交换机到另一个网卡，  10Gb/s

### deepspeed的加速

隐藏 ， 流水线， 计算第一层同时从cpu取第二层参数， 8月3日1.5B 十五亿可以。 阿里最顶尖的机器， 8张v100。

只到SSD , 我们是普通磁盘HDD,不支持， 

GPU 的参数缓存offload到内存。  

采用不同technology 画出几条反函数曲线。  每条 。 

先不开offload 跑

华为盘古， 阿里和清华合作， 做了上百亿参数的模型。 

有的训练用一万张v100.  训练一次一千万美元。 

有哪些optimizetion， 一个个打开。

1. "gradient_accumulation_steps": 1, 

先求batch size = 4的， 得到gradient ，然后再算一个4 ， 这样相当于 batchsize  = 4。     有时候显存不够就必须把他调大。 可能和别的冲突。

2. fp16  ， 就是gpu计算时 降低精度到fp16 .  cuda 有加速fp16的硬件。 

优化方法： checkpoint， fp16，  offload

3. offload是 把中间结果 gpu上没用到的中间结果 swap到cpu上， 用到再取回来。

4. checkpoint是 中间结果占据了显存， 我们丢掉中间结果，需要用到的时候， 从第六个tensor 开始算 ，也就是从他最近的那个checkpoint 重新计算。  时间换空间。 
5. zero stage 1 2 3     ，stage =3的时候会慢， 把parameter swap 到内存上， 可以训练很大的模型 。   google 把训练好的bert给大家，普通人没有这么大的算力来预训练， 大家fine tune 一下就可以了。



输入weight和activation ，GPU 可以存下这个变量就可以训练了。  



8个GPU 分一层， 一个GPU 有一层的1/8 . GPU 显存可以   容纳一层的 1/8 是极限了. 

conv 计算量大， 就checkpoint， 重算计算量小的但是占据显存大的relu和pool层。 

每个 GPU 只有16G 显存， 放不下所有参数， 所以参数更新是到CPU上。 



HDD磁盘交换速度很慢。 需要NVME磁盘。

实现方式： 

1. 梯度累积是  forward， backward， 最后update
2.  混合精度fp16， torch 可以直接 mix 混合精度
3. torch每一层提供了hook ， 可以fork to cuda ， fork to cpu 。
4. 

本次我们实验使用的机器配置磁盘空间不足. 

本次项目实践中由于阿里云的硬盘不支持, 所以没能实践. 



1.本项目的创新点在哪里？

找出了pytorch大模型训练瓶颈

量化了各显存优化技术对训练大模型的贡献度

找出了单机八卡训练模型参数的极限值. 

2.本项目的意义是什么？

可以训练巨大模型， 实现更多任务。 华为, 清华北大,谷歌大脑,openai都在做. 





我们大参数很慢， 但是fine-tuning不需要多快。 



本来 150亿参数需要多大？   pytorch  DDP  单机7亿。 

极限怎么算出来？ 



混合精度训练

fp16 参数， fp16的梯度， fp32的参数，fp32的梯度，  fp32优化器里的状态比如动量和方差（默认fp32可以开 fp16），  一个参数对应20个字节。  15billion 参数。  

activation 就是 激活显存。 



### 第一组约课

[二分图的最大匹配、完美匹配和匈牙利算法 - Blog - Renfei Song](https://www.renfei.org/blog/bipartite-matching.html)

简单来说，如果图中点可以被分为两组，并且使得所有边都跨越组的边界，则这就是一个二分图。准确地说：把一个图的顶点划分为两个不相交集 UU 和VV ，使得每一条边都分别连接UU、VV中的顶点。如果存在这样的划分，则此图为一个二分图。二分图的一个等价定义是：不含有「含奇数条边的环」的图。图 1 是一个二分图。为了清晰，我们以后都把它画成图 2 的形式。

二分图最大匹配 dinic 算法， 利用网络流解决二分图最大匹配。

**匹配**：在图论中，一个「匹配」（matching）是一个边的集合，其中任意两条边都没有公共顶点。

**最大匹配**：一个图所有匹配中，所含匹配边数最多的匹配，称为这个图的最大匹配。图 4 是一个最大匹配，它包含 4 条匹配边。 

1. 有向图连接反向边
2. 图分层



### 第二组

匹配阈值算法，  好厉害， 视频会议 识别是谁讲话， 多人讲话语音识别。 

如何提升效果？

1. 场景更加匹配的数据集
2. 测试数据少， 不能反映普遍规律， 增加数据集。
3. 





### 第八组



不同 降级框架的



每个广告位求平均 score

优先drop 广告位平均score 最小流量 rank。



lightGBM算法 

线性回归

MLP



可以对不同的流量分配不同复杂程度的模型， sim， din ，dien



### 第九组

low-light image

暗视频增强。

亮度增强， 色彩调节， 噪声去除， 时序稳定化。 

 severe color cast

over-exposition  过曝

visually pleasing results

输入比较暗， 调高，输入不太暗就不调高太多。 


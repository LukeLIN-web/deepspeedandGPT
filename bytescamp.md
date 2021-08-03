

有现金奖励

### mentor的背景:

AML ,  高性能 , io 加速,  yi bairen 和 pengyanghua 彭杨华

Applied Machine Learning

## poster

夏令营组委会希望大家能在8月2日18:00前提交poster内容，用于8月5日游园会，组委会统一制作展板，ppt中是for营员的制作要求

划重点了：提交展板制作内容的ddl是【8月2日18:00前】，若项目还未产出结果，建议大家可以用【预期实现内容、预期产出结果】替代，具体展示内容不做要求（eg：项目摘要、技术原理、项目亮点等），主要是满足不同方向营员的交流与学习~"

### 队友

李想,  北邮本硕,  物流工程,  马上研二 微软亚洲研究院和腾讯实习,ACM,  报名了调度算法,  调剂到 高性能计算.  它投了其他实习流程但是被结束了. 

李想  计算所马上研二,  本科南航, 都是计算机 ,  cache 切换算法.disk上多级cache . 主要系统的背景.    

笑死, 彭杨华说西瓜书我们还没看过.

### 这个项目

GPT   transformer  序列- 序列. 给定一个文本序列, 生成后面的文本或者翻译文本.   表示序列-序列的过程, 



好像就是 面试可以少一次面试.  

下周一再同步一次. 

多卡, deepspeed , GPT. 







数据并行有什么不同的地方? 

https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/ 可以看这个博客里面的动画 学习



https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/ 这个是需要应用到GPT2的训练中.  应该要先复现. 一开始层太多, 可以调小层数.  

我们不使用真实数据, 下载太慢了.  和fake data 测试也差不多.  GPT pretaining https://github.com/NVIDIA/Megatron-LM#gpt-pretraining

``` 
--num-layers 24  可以改成4 ,之后慢慢往上调.
text_document, vocab-file $VOCAB_FILE \
          --merge-file 这些都可以用假数据. 
          准备数据. 
```

Data Preprocessing 里面就有 教你怎么生成假数据. The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. 

尝试运行这个https://github.com/microsoft/DeepSpeedExamples/tree/25d73cf73fb3dc66faefa141b7319526555be9fc/Megatron-LM-v1.1.5-ZeRO3

先看两个blog, 然后运行这个. 





8.1



8.2

[Megatron-LM GPT2 - DeepSpeed](https://www.deepspeed.ai/tutorials/megatron/) 

这个已经跑通了. 

要把数据下载到 `DeepSpeedExamples/Megatron-LM/data`:



计算参数量. 






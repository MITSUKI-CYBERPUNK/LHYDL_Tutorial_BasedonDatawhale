# 7 Transformer
## 7.1 序列到序列模型
+ 序列到序列模型输入和输出都是一个序列，输入与输出序列长度之间的关系有两种情况。
	+ 第一种情况下，输入跟输出的长度一样；
	+ 第二种情况下，机器决定输出的长度。

### 7.1.1 语音识别，机器翻译与语音翻译
![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523194854.png)
+ 有很多语言没有文字，无法做语音识别，因此需要做语音翻译，直接翻译成文字。

### 7.1.2 语音合成
+ 输入文字、输出声音信号就是语音合成（Text-To-Speech，TTS）
### 7.1.3 聊天机器人
![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523201430.png)

### 7.1.4 问答任务
+ 序列到序列模型在自然语言处理的领域的应用很广泛，而很多自然语言处理的任务都可以想成是问答（Question Answering，QA）的任务，比如下面是一些例子：
	+ 翻译
	+ 自动做摘要
	+ 情感分析
+ **问答就是给机器读一段文字，问机器一个问题，希望它可以给出一个正确的答案。**
+ 序列到序列模型就像瑞士刀，瑞士刀可以解决各式各样的问题，砍柴可以用瑞士刀，切菜也可以用瑞士刀，但是它不一定是最好用的。

### 7.1.5 句法分析
+ **句法分析（syntactic parsing）**：给机器一段文字：比如“deep learning is very powerful”，机器要产生一个句法的分析树，即句法树（syntactic tree）。通过句法树告诉我们 deep 加 learning 合起来是一个名词短语，very 加 powerful 合起来是一个形容词短语，形容词短语加 is 以后会变成一个动词短语，动词短语加名词短语合起来是一个句子。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523201741.png)
+ 在句法分析的任务中，输入是一段文字，输出是一个树状的结构，而一个树状的结构可以看成一个序列，该序列代表了这个树的结构，如图所示。把树的结构转成一个序列以后，我们就可以用序列到序列模型来做句法分析。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523201849.png)

### 7.1.6 多标签分类
+ 多标签分类（multi-label classification）任务也可以用序列到序列模型。多类的分类跟多标签的分类是不一样的。如图所示，在做文章分类的时候，同一篇文章可能属于多个类，文章 1 属于类 1 和类 3，文章 3 属于类 3、9、17。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523202009.png)

+ **多分类问题（multi-class classification）是指分类的类别数大于 2。而多标签分类是指同一个东西可以属于多个类。**
+ 多标签分类问题不能直接把它当作一个多分类问题的问题来解。比如把这些文章丢到一个分类器里面，本来分类器只会输出分数最高的答案，如果直接取一个阈值（threshold），只输出分数最高的前三名。这种方法是不可行的，因为每篇文章对应的类别的数量根本不一样。因此需要用**序列到序列模型**来做，如图所示，输入一篇文章，输出就是类别，机器决定输出类别的数量。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523202519.png)

+ 另外，目标检测也可以使用S2S模型来做

## 7.2 Transformer结构
+ 一般的序列到序列模型会分成编码器和解码器，如图所示。**编码器负责处理输入的序列，再把处理好的结果“丢”给解码器，由解码器决定要输出的序列。**
+ ![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523202744.png)

+ Transformer的详细结构如下：![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523203341.png)

## 7.3 Transformer编码器
+ 编码器输入一排向量，输出另外一排向量。自注意力、循环神经网络、卷积神经网络都能输入一排向量，输出一排向量。Transformer的编码器使用的是**自注意力**，输入一排向量，输出另外一个同样长度的向量。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523203159.png)

+ 如图所示，编码器里面会分成很多的**块（block）**，每一个块都是输入一排向量，输出一排向量。输入一排向量到第一个块，第一个块输出另外一排向量，以此类推，最后一个块会输出最终的向量序列。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523203411.png)
+ 在 Transformer 的 Encoder 部分，有 N 个 Blocks，每个 Block 都是由 Self-attention（图中“Multi-Head Attention” 单元）和 Fully Connected Layer（图中 “Feed Forward” 单元）组成。![v2-59b4e252fcc79c9db54100366a6abbb4_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-59b4e252fcc79c9db54100366a6abbb4_1440w.webp)

+ Transformer 里面加入了**残差连接（residual connection）** 的设计，如所示，最左边的向量 b 输入到自注意力层后得到向量 a，输出向量 a 加上其输入向量 b 得到新的输出。得到残差的结果以后，再做**层归一化（layer normalization）**。层归一化比信念网络更简单，不需要考虑批量的信息，而批量归一化需要考虑批量的信息。层归一化输入一个向量，输出另外一个向量。层归一化会计算输入向量的平均值和标准差。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523204014.png)

+ Batch Normalization 是对一个 batch 的 sequences，在 vector 的维度上求均值方差后做 Normalization。 对于 Self-attention， 不同的输入 sequences ，长度不同。当输入 sequence 长度变化大时，不同 batch 求得的均值方差抖动大。此外，如果测试时遇到一个很长的 sequence（超过训练集中的任何 sequence 长度），使用训练时得到的均值方差可能效果不好。而 Layer Normalization 是在每个 sample 上做，不受 sequence 长度变化的影响，所以这里用的是 **Layer Normalization**。Bert的解码器也与Transformer类似。
## 7.4 Transformer解码器
### 7.4.1 自回归解码器AT (Autoregressive)
+ Autoregressive 是什么意思呢？就是指**前一时刻的输出，作为下一时刻的输入。** 如下图所示，起始时要输入一个特别的 Token（图中的“Start”），告诉 decoder：一个新的 sequence 开始了！这个 Token 经过 Decoder，输出一个 vector，里面是各类别的概率，从中选出最大概率对应的类别，如图示例子的“机”字，作为输出。“机” 字对应的变量又作为第二个输入，经过 Decoder 得到第二个输出：“器”，依此类推。![v2-6177c3e822fb37a5eb28f37c3361ee56_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-6177c3e822fb37a5eb28f37c3361ee56_1440w.webp)

+ 再来看看 decoder 的内部结构。先把 decoder 和 encoder 放在一起对比看看，如下图所示，除去灰色方框遮住的部分，decoder 和 encoder 非常相似。但是为什么 decoder 用的是 **Masked Multi-Head Attention**？![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523204959.png)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523205025.png)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523205233.png)


+ 还有一个关键的问题，输出何时终止？
	+ 解决办法：加入 **Stop Token**。输入最后一个字符时，输出 “END”，此时机器就知道输出 sequence 完成了。
	+ 如图 7.23 所示，要让解码器停止运作，需要特别准备一个特别的符号 <EOS>。产生完“习”以后，再把“习”当作解码器的输入以后，解码器就要能够输出 <EOS>，解码器看到编码器输出的嵌入、<BOS>、“机”、“器”、“学”、“习”以后，其产生出来的向量里面 <EOS> 的概率必须是最大的，于是输出 <EOS>，整个解码器产生序列的过程就结束了。
+ ![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523205614.png)


### 7.4.2 非自回归解码器NAT (Non-autoregressive)
+ 下图所示分别为 AT (autoregressive) 与 NAT (Non-autoregressive) 的结构示意图。可以看到，与 AT 不同的是，NAT 并不使用之前时刻的输出，而是一次输入一组 special token。那么，输入多长合适呢？有多种方法。比如，一种方法是把 encoder 的输出送入一个 Classifier，预测 decoder 输出 sequence 长度，进而也输入相同长度的 special token。另一种方法如下图所示，输入一个很长的 sequence，看输出什么时候出现 stop token，截取其之前的 sequence 作为输出。![v2-586c525ef68e3a537f825627fee44e1f_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-586c525ef68e3a537f825627fee44e1f_1440w.webp)

+ 这两者各有利弊：AT 一次输出一个 vector（因为上一个输出又作为下一个输入），无法并行处理。而 NAT 可以并行处理。NAT 是在 transformer 提出 self-attention 之后才出现的。为什么呢？上一节课提到，RNN 也是对 vector 一个一个地处理，没办法做并行。self-attention 才是并行处理，一次处理一个 sequence。

+ 此外，NAT 可以调节输出 sequence 长度。比如在语音合成 (TTS) 任务中，按前面提到的方法一，把 encoder 的输出送入一个 Classifier，预测 decoder 输出 sequence 长度。通过改变这个 Classifier 预测的长度，可以调整生成语音的语速。例如，设置输出 sequence 长度 x2，语速就可以慢一倍。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523210013.png)

## 7.5 编码器-解码器注意力
+ 编码器和解码器通过编码器-解码器注意力（encoder-decoder attention）传递信息，编码器-解码器注意力是连接编码器跟解码器之间的桥梁。如图所示，解码器中编码器-解码器注意力的键和值来自编码器的输出，查询来自解码器中前一个层的输出。
+ 此时，我们来看刚才灰色方块遮住的部分：Cross attention，也就是下图中红色方框部分。它计算的是 encoder 的输出与当前 vector 的 cross attention 。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523210511.png)
+ 具体操作为：用 decoder 中 self attention 层的输出 vector 生成q，与由 encoder 最后一层输出 sequence 产生的(k,v)做运算，如下图所示：![v2-7bf9bca9d768f0adeef0fa872dce9ad2_1440w.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-7bf9bca9d768f0adeef0fa872dce9ad2_1440w.png)

+ 试想一下，如果你要做一个演讲，虽然记了演讲内容，但还是会担心一上台紧张忘词。怎么办呢？可以把提纲写在卡片上，演讲中看上一眼，就知道接下来要讲什么了。我觉得 cross attention 的作用就有点像这个小卡片，因为它看过整个 sequence，可以给 decoder 提示信息。

## 7.6 Transformer的训练过程
+ **训练过程**：
	+ 如下图所示，decoder 的输出 (output) 是一个概率分布，label 是 one-hot vector，优化的目标就是使 label 与 decoder output 之间的 cross entropy 最小。这其实是一个分类问题。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523211838.png)
+ **教师强制**：
	+ 前面介绍，decoder 中，前一个输出又作为下一个输入。使用 Teacher Forcing 方法，decoder 输入用的是 ground truth value。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523211952.png)

## 7.7 序列到序列模型训练常用技巧
### 7.7.1 复制机制Copy Mechanism
+ 有些情况，不需要对输入做改动，比如翻译人名地名，聊天机器人(chat-bot)，摘要 (summarization) 等，可以直接复制一部分输入内容。
+ 这里说说 chat-bot 中使用 Copy Mechanism 的情况。比如，打招呼。用户说：“你好，我是_宁萌_。” 机器回：“_宁萌_ 你好！”。再比如：复述疑问，也就是机器“不明白”/不知道的地方。用户说：“你觉得 _《三体》_ 怎么样？”机器回：“ _《三体》_ 是什么？”

+ 具体的方法：Pointer Network , copy network

### 7.7.2 引导注意力Guided Attention
+ 李老师首先举了一个语音合成 (TTS) 的例子，机器一次说四遍“发财”这个词时，说得挺好，还有抑扬顿挫之感。一次说三遍或两遍“发财”也正常。但是，一次说一遍“发财”时，不知怎的，只有一个音“财”。

+ 从这个例子可以看到，在处理语音识别 (speech recognition) 或语音合成 (TTS)等任务时，我们不希望漏掉其中的任何一段内容，Guided Attention 正是要满足这个要求。而 chat-bot, summary 一类的应用在这方面的要求就宽松得多。

+ Guided Attention 是让 attention 的计算按照一定顺序来进行。比如在做语音合成时，attention 的计算应该从左向右推进，如下图中前三幅图所示。如果 attention 的计算时顺序错乱，如下图中后三幅图所示，那就说明出了错误。具体方法：Monotonic Attention, Location-aware attention。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523212654.png)

### 7.7.3 束搜索Beam Search
+ 这其实是一个最优路径的问题。前面介绍，decoder 每次输出一个变量，假设输出词汇库只有 A, B 两个词汇。每一次都选择最大概率的作为输出，如下图中红色路径所示，这就是 Greedy Decoding。同时，decoder 的每个输出又是下一时刻输入，如果我们从整个 sequence 的角度考虑，可能第一次不选最大概率，后面的输出概率（把握）都很大，整体更佳，如下图中绿色路径所示。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240523212822.png)
+ 束搜索经常也称为集束搜索或柱搜索。束搜索
是用比较有效的方法找一个近似解，在某些情况下效果不好。比如论文“The Curious Case OfNeural Text Degeneration”[5]。这个任务要做的事情是完成句子（sentence completion），也就是机器先读一段句子，接下来它要把这个句子的后半段完成，如果用束搜索，会发现说机器不断讲重复的话。如果不用束搜索，加一些随机性，虽然结果不一定完全好，但是看起来至少是比较正常的句子。有时候对解码器来说，没有找出分数最高的路，反而结果是比较好的，这个
就是要看任务本身的特性。假设任务的答案非常明确，比如语音识别，说一句话，识别的结果就只有一个可能。对这种任务而言，通常束搜索就会比较有帮助。但如果任务需要机器发挥一点创造力，束搜索比较没有帮助。

### 7.7.4 加入噪声
+ 在做语音合成的时候，解码器加噪声，这是完全违背正常的机器学习的做法。在训练的时候会加噪声，让机器看过更多不同的可能性，这会让模型比较鲁棒，比较能够对抗它在测试的时候没有看过的状况。但在测试的时候居然还要加一些噪声，这不是把测试的状况弄得更困难，结果更差。但语音合成神奇的地方是，模型训练好以后。测试的时候要加入一些噪声，合出来的声音才会好。用正常的解码的方法产生出来的声音听不太出来是人声，产生出比较好的声音是需要一些随机性的。对于语音合成或句子完成任务，解码器找出最好的结果不一定是人类觉得最好的结果，反而是奇怪的结果，加入一些随机性的结果反而会是比较好的。

### 7.7.5 使用强化学习训练Optimizing Evaluation Metrics
+ 在 homework 中，train 使用 cross entropy loss 做 criterion，要使 output 和 label 在对应 vector 上 cross-entropy 最小。而评估模型用的是 BLEU score, 是在两个 sequence 上运算，如下图所示。因此，validation 挑选模型时也用 BLEU score 作为衡量标准。![v2-0802f9362b5b2f949044a42ccb18d0c1_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-0802f9362b5b2f949044a42ccb18d0c1_1440w.webp)

+ 那么， train 直接就用 BLEU score 做 criterion 岂不更好？
	+ 问题就在于：BLEU score 没办法微分，不知道要怎么做 gradient descent。实在要做：Reinforcement Learning(RL)。

+ **秘诀：”When you don’t know how to optimize, just use reinforcement learning(RL).” 遇到在 optimization 无法解决的问题，用 RL “硬 train 一发”。**

### 7.7.6 计划采样Exposure bias
+ 训练时 Decoder 看的都是正确的输入值（ Ground Truth ）。测试时如果有一个输出有错误，可能导致后面都出错。
+ **解决办法**：训练时 decoder 加入一点错误的输入，让机器“见识” 错误的情况，这就是 Scheduling sampling。



+ 如果说之前的 RNN 模型还在模仿人类的语言处理方式，逐个字词地理解，那么 Self-attention 则是充分利用了机器的特点，并行处理信息，达到了“一目十行”的效果。不知怎的，我想到了 Ted Chiang （特德姜 ）写的科幻小说 “Story of Your Life”（《你一生的故事》），里面的外星人语言就是团状的，不像我们人类的语言是线性时序的，因此外星人可以预知未来。而地球上的一位科学家在交流中学会了这种语言，由此也提前知道了自己以后的人生。也就是说，语言影响思维方式。当然，这只是小说中的想象哈哈。

+ 目前是机器在学习人类的交流方式，我们人类说话写字都是时序的，有前后顺序。transformer 的 self-attention 虽然很强大，但没有考虑位置信息，因此需要额外加入 positional encoding。未来有没有类似科幻小说设想的可能，人类学习机器并行处理信息的方式，在语言交流中减少时序信息，因而更加高效
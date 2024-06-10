# 6 自注意力机制Self-Attention Model
+ 自注意力模型也是一个常见的网络架构。首先我们要知道，**输入可以看作一个向量**：回归问题输出为标量，分类问题输出为类别。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520194754.png)

## 6.1 输入是向量序列的情况
+ **输入是一组向量，且输入的向量的数量会改变，即每次模型输入的序列长度都不一样**，我们举例来说明如何处理：![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520195003.png)
	+ **文字处理**：假设网络的输入是一个句子，每一个句子的长度都不一样（每个句子里面词汇的数量都不一样）。如果把一个句子里面的每一个词汇都描述成一个向量，用向量来表示，模型的输入就是一个向量序列，而且该向量序列的大小每次都不一样。
		+ 将词汇表示成向量最简单的做法是**独热编码**，但它假设所有词汇之间没有关系，其中没有任何语义信息。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520195455.png)
		+ **词嵌入(word embedding)** 也可将词汇表示成向量，它用一个向量来表示一个词汇，而这个向量是包含语义信息的。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520195638.png)如果把词嵌入画出来，所有的动物可能聚集成一团，所有的植物可能聚集成一团，所有的动词可能聚集成一团等等。词嵌入会给每一个词汇一个向量，而一个句子就是一组长度不一的向量。
	+ **声音信号**：一段声音信号其实是一组向量。我们会把一段声音信号取一个范围，这个范围叫做一个**窗口（window）**，把该窗口里面的信息描述成一个向量，这个向量称为**一帧（frame）**。通常这个窗口的长度就是 25 毫秒。为了要描述一整段的声音信号，我们会把这个窗口往右移一点，通常移动的大小是 10 毫秒。(前人经验，最佳选择)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520200048.png)
	+ **图**：一个图（graph）也是一堆向量。社交网络是一个图，在社交网络上面每一个节点就是一个人。每一个节点可以看作是一个向量。每一个人的简介里面的信息（性别、年龄、工作等等）都可以用一个向量来表示。

### 6.1.1 类型1：输入与输出数量相同
+ 输入n个向量，输出n个标签![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520201249.png)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520200724.png)
	+ 在文字处理上，假设我们要做的是**词性标注（Part-Of-Speech tagging，POS tagging）**。机器会自动决定每一个词汇的词性，判断该词是名词还是动词还是形容词等等。例如说：I saw a saw.
	+ 如果是语音，一段声音信号里面有一串向量。每一个向量都要决定它是哪一个音标。这不是真正的语音识别，这是一个语音识别的简化版。
	+ 如果是社交网络，给定一个社交网络，模型要决定每一个节点有什么样的特性，比如某个人会不会买某个商品，这样我们才知道要不要推荐某个商品给他。

### 6.1.2 类型2：输入是一个序列，输出是一个标签
+ 输入是一个序列，输出是一个标签![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520201108.png)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520201225.png)
	+ 情感分析就是给机器看一段话，模型要决定说这段话是积极的（positive）还是消极的（negative）
	+ 如果是语音，机器听一段声音，再决定是谁讲的这个声音。
	+ 如果是图，比如给定一个分子，预测该分子的亲水性。

### 6.1.3 类型3：序列到序列
+ 我们不知道应该输出多少个标签，机器要自己决定输出多少个标签。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520201353.png)
	+ 翻译就是序列到序列的任务，因为输入输出是不同的语言，它们的词汇的数量本来就不会一样多。
	+ 真正的语音识别输入一句话，输出一段文字，其实也是序列到序列的任务。

## 6.2 自注意力运作原理
+ 先讲输入跟输出数量一样多的状况，以序列标注为例。
	+ 如果使用普通的FC，无法区分不同词性相同的词：![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520202136.png)
	+ 将每个向量的前后几个向量串起来一起输入到全连接网络便可：![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520202225.png)
	+ 在语音识别里面，我们不是只看一帧判断这个帧属于哪一个音标，而是看该帧以及其前后 5 个帧（共 11 个帧）来决定它是哪一个音标。所以可以给全连接网络一整个窗口的信息，让它可以考虑一些上下文，即与该向量相邻的其他向量的信息。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520202354.png)
	+ 但是某些任务不能只考虑一个窗口，而是要考虑一整个序列。然而设置全长窗口由于每次序列长度的不同而变得很麻烦(因为需要统计训练的数据，设置很大的窗口，需要更多的参数，运算量大且容易过拟合)
+ **自注意力模型的引入**：![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240520204817.png)
	+ 自注意力模型会“吃”整个序列的数据，输入几个向量，它就输出几个向量。而这 4 个向量都是考虑整个序列以后才得到的，所以输出的向量有一个黑色的框，代表它不是一个普通的向量，它是考虑了整个句子以后才得到的信息。接着再把考虑整个句子的向量丢进全连接网络，再得到输出。因此全连接网络不是只考虑一个非常小的范围或一个小的窗口，而是考虑整个序列的信息，再来决定现在应该要输出什么样的结果，这就是自注意力模型。
	+ 自注意力模型可以和全连接层多次交替使用：**全连接网络专注于处理某一个位置的 信息，自注意力把整个序列信息再处理一次。**![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521153122.png)
+ **自注意力模型的运作过程**：
	+ 目的是**寻找 vector 之间的关系**。如下图所示，对于某一个 vector，求出其与 sequence 中其它 vectors 的相关性。我们可以用$alpha$来表示相关性。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521161159.png)![v2-25787f55d647627c4b524b9ffd655841_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-25787f55d647627c4b524b9ffd655841_1440w.webp)
	+ 如何计算向量关联度$alpha$:用**点乘Dot-product**或者**加法Additive**都可以，前者比较常见：![v2-77a230bf397ad6a41060ab0246514d4b_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-77a230bf397ad6a41060ab0246514d4b_1440w.webp)
		+ **Dot-product**：（更常用也是transformer中的方法）把输入的两个向量分别乘上两个不同的矩阵，左边这个向量乘上矩阵Wq，右边这个向量乘上矩阵Wk，得到两个向量q跟k，再把q跟k做点积，把它们做逐元素（element-wise）的相 乘，再全部加起来以后就得到一个标量（scalar）α
		+ **Additive**：把两个向量通过Wq、Wk得到q和k，但不是把它们做点积，而是把 q 和k“串”起来“丢”到一个tanh函数，再乘上矩阵W 得到α。
	+ 套用在自注意力模型里：
		+ 自注意力模型一般采用**查询-键-值（Query Key-Value，QKV）模式**
		+ ![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521162333.png)
		+ ![v2-8502ffb756770a5ca3057b9dd08f52cc_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-8502ffb756770a5ca3057b9dd08f52cc_1440w.webp)
			+ **Softmax操作**：ReLu也可，有时更好
		+ ![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521162958.png)![v2-3ec548e66ec3c08c9df0ea5739f99825_1440w.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-3ec548e66ec3c08c9df0ea5739f99825_1440w.png)![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521163319.png)
	+ **从个体转换到整体**：输入是 vector set (sequence)：![v2-cbb1fdebe5353fd6e0e415293c562947_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-cbb1fdebe5353fd6e0e415293c562947_1440w.webp)
	+ 将k转置与q相乘，得到对应的alpha，组成注意力矩阵A：![v2-eb5273fe9af681a161e4aefc3cb59fd8_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-eb5273fe9af681a161e4aefc3cb59fd8_1440w.webp)
	+ a‘与v相乘得到输出b，也用矩阵表示:![v2-1abdaaafde69861ba0d7a577b4f185c1_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-1abdaaafde69861ba0d7a577b4f185c1_1440w.webp)
	+ 综上，自注意力全流程矩阵运算表达式：![v2-ea387546eca3eed6d86ff93725f0ace0_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-ea387546eca3eed6d86ff93725f0ace0_1440w.webp)

## 6.3 多头注意力Multi-head Self-attention
+ 在使用自注意力计算相关性的时候，就是用q去找相关的k。但是有**多种相关性**，所以也许可以有多个q，不同的q负责不同种类的相关性，这就是多头注意力。此时，（q，k，v）由一组变成多组。**注意：每组的（q，k，v）是对应运算的，不跨组。**![v2-c633d5a6f68d6881d3404e948e915742_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-c633d5a6f68d6881d3404e948e915742_1440w.webp)
+ 与单个的 self attention 相比，Multi-head Self-attention 最后多了一步：由多个输出组合得到一个输出。![v2-bbc36e29d8e4fd3c2ff47f7615f4f855_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-bbc36e29d8e4fd3c2ff47f7615f4f855_1440w.webp)
+ 下图所示为句子 "The animal didn't cross the street because it was too tired." 中 "it" 一词的 2-head attention。这两种 attention 分别用橙色和绿色表示，颜色越深，表示 attention 越大。可以看到，其中一个 head 的 attention 集中在 "The animal" ，另一个 head 的 attention 集中在 "tired"。这也正好描述了 "it" 在句子中的相关性，一是其指代 the animal，二是它的状态 tired。![v2-b8b0f4810397ba1f120b03f1c34a40e8_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-b8b0f4810397ba1f120b03f1c34a40e8_1440w.webp)

## 6.4 位置编码Positional Encoding
+ 回顾 self attention 的计算过程，我们发现 self-attention 没有考虑位置信息，只计算互相关性。比如某个字词，不管它在句首、句中、句尾， self-attention 的计算结果都是一样的。但是，有时 Sequence 中的位置信息还是挺重要的。![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240521164519.png)![v2-0fa5043f02ce2d1865f23da14ccf90d5_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-0fa5043f02ce2d1865f23da14ccf90d5_1440w.webp)

## 6.5 截断自注意力Truncated Self-attention
+ 将自注意力用于语音处理时，可以对自注意力做一些小小的改动
+ **截断自注意力（truncated self-attention）** 可以处理向量序列长度过大的问题。截断自注意力在做自注意力的时候不要看一整句话，就只看一个小的范围就好，这个范围是人设定的。可以加速处理，避免序列过长导致资源占用或无法处理。（Truncated Self-attention 感觉有点像 CNN 的 receptive field。）![v2-206d6bcbf41526bdc6da480a4f93ac90_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-206d6bcbf41526bdc6da480a4f93ac90_1440w.webp)

## 6.6 vs CNN in CV
+ 如图所示，把一个像素点（W,H,D）当成一个 vector，一幅图像就是 vector set。![v2-6bf55122520a9a8a4a3c80c196669eeb_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-6bf55122520a9a8a4a3c80c196669eeb_1440w.webp)
+ Self-attention 可以看成是更加灵活的 CNN。
	+ 把一个像素点当作一个 vector，CNN 只看 receptive field 范围的相关性，可以理解成中心的这个 vector 只看其相邻的 vectors，如下图所示。从 Self-attention 的角度来看，这就是在 receptive field 而不是整个 sequence 的 Self-attention。因此， CNN 模型是简化版的 Self-attention。
	+ 另一方面，CNN 的 receptive field 的大小由人为设定 ，比如: kernel size 为 3x3。而 Self-attention 求解 attention score 的过程，其实可以看作在学习并确定 receptive field 的范围大小。与 CNN 相比，self-attention 选择 receptive field时跳出了相邻像素点的限制，可以在整幅图像上挑选。因此，Self-attention 是复杂版的 CNN 模型。
	+ ![v2-616f2ccbf85b7024ba14799fcc81808d_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-616f2ccbf85b7024ba14799fcc81808d_1440w.webp)通过上面的分析可以发现，self-attention 复杂度更大 (more flexible)，|H| 更大，因此需要训练集的数据量 N 更大。如下图所示，在图像识别 (Image Recognition) 任务上，数据量**相对**较小 (less data) 时，CNN 模型表现更好。数据量相对较大 (more data) 时，Self-attention 模型表现更好。![v2-ed0a7715f73bc93d059fe21abf9bbc6c_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-ed0a7715f73bc93d059fe21abf9bbc6c_1440w.webp)
+ 我们可以说，CNN是Self Attention的一个子集（conformer里面同时用到了自注意力和CNN）
## 6.7 vs RNN when input is Seq
+ Bidirectional RNN双向RNN可从两个方向对句子分析，具体区别其实是：
	+ 如下图所示，如果RNN 最后一个 vector 要联系第一个 vector，比较难，需要把第一个 vector 的输出一直保存在 memory 中。而这对 self-attention 来说，很简单。整个 Sequence 上任意位置的 vector 都可以联系，“**天涯若比邻**”，距离不是问题。
	+ RNN 前面的输出又作为后面的输入，因此要依次计算，无法并行处理。 self-attention 可以**并行计算**。![v2-c22a4528506e879c56f69b0127d5f337_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-c22a4528506e879c56f69b0127d5f337_1440w.webp)

## 6.8 图与自注意力
+ Graph 中，可以根据 edge 来简化 attention 计算。有 edge 连接的 nodes 就计算 attention，没有 edge 连接的就设为 0，这是 Graph Neural Network(GNN) 的一种。![v2-c4fcba324557a9822f8ab4f28b60215b_1440w.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/v2-c4fcba324557a9822f8ab4f28b60215b_1440w.webp)

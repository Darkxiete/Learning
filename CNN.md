# 卷积层

## 整体把握

$$
W_2=1+\frac{W_1-W_f+2*P}{S} \\
其中W_f为filter的宽度，S为步长
$$

在处理彩色图像时，输入尺寸为$W*H*D$，$filter$尺寸为$m*m*D$，卷积的过程就是从输入的左上角，按照步长$S$从左往右，从上往下移动，用输入*$filter$然后加上$bias$，得到相应的点
$$
ret_{i,j}=\sum_{k=1}^{D}Matrix_{input_k}*filter_k+bias
$$
之所以有填充P（padding），是因为对于特定的步长S，存在有部分输入数据无法与$filter$相乘的问题，加上P之后就可以取到。如下图所示

![padding](pic/padding.png)

## 计算过程

整个流程图如下图所示。

![conv_layer](pic/conv_layer.png)

![conv](pic/conv.gif)



## 特征图大小计算

输入大小为：$W_1 \times H_1 \times D_1$

需要指定的超参数：$filter个数(K)，filter大小(F)，步长(S)，边界填充(P)  $

输出：
$$
\begin{align}
W_2&=(W_1-F+2\times P)/S +1\\
H_2&=(H_1-F+2\times P)/S +1\\
D_2&=K
\end{align}
$$
![conv_layer_example](pic/conv_layer_example.png)

# 激活函数

## 相关问题

1. 为什么要使用激活函数？
   1. 激活函数是用来加入非线性因素的，提高神经网络对模型的表达能力，解决线性模型所不能解决的问题。 
2. 有哪些常用的激活函数？
3. 不同激活函数的特点是什么？解决了什么问题？
4. 常见问题有哪些？
   1. outputs are not zero-centered，会导致在权值更新时，产生[约束性捆绑问题](https://www.zhihu.com/question/57194292)
   2. gradients vanish. 
5. 梯度消失，梯度爆炸问题是什么？为什么？怎么解决？
   1. 预训练（逐步训练）+微调，原理就是逐步得出多个局部最小，然后一起求全局最小
   2. 梯度剪切、正则
   3. 使用ReLU,PReLU,RReLU等激活函数
   4. [Batchnorm](https://blog.csdn.net/qq_25737169/article/details/79048516),通过对每一层的输出规范为均值和方差一致的方法，消除了ww带来的放大缩小的影响，进而解决梯度消失和爆炸的问题，或者可以理解为BN将输出从饱和区拉倒了非饱和区。加入后，CNN结构为：  $conv->bn->scale->relu $
   5. 引入残差结构，比如[ResNet](https://zhuanlan.zhihu.com/p/31852747)



> [知乎专栏](https://zhuanlan.zhihu.com/p/25110450)
>
> [激活函数](https://blog.csdn.net/elaine_bao/article/details/50810598)
>
> [激活函数](https://blog.csdn.net/cyh_24/article/details/50593400)
>
> [梯度爆炸和梯度消失及解决方案](https://blog.csdn.net/qq_25737169/article/details/78847691)：从深层网络角度来讲，不同的层学习的速度差异很大，表现为网络中靠近输出的层学习的情况很好，靠近输入的层学习的很慢，有时甚至训练了很久，前几层的权值和刚开始随机初始化的值差不多。 



一般CNN采用的激活函数是ReLU（The Rectified Linear Unit/修正线性单元 ），添加这一层的原因是为了加入非线性。



## 无激励函数的神经网络

对于无激活函数的神经网络而言，其常见情况如下：

1）单层感知机![perception_1](pic/perception_1.png)

2）多个感知机![perception_2](pic/perception_2.png)

3）多层感知机

![perception_3](pic/perception_3.png)

> **没有激励函数的神经网络的输出是线性方程，其在用复杂的线性组合来逼近曲线** 

## 有激励函数的神经网络

引入非线性的激活函数之后，每一层输出都加上了非线性因素。

![perception_4](pic/perception_4.png)

![perception_5](pic/perception_5.png)



![perception_6](pic/perception_6.png)

## 常用的激活函数讨论

常见的激活函数有

1. $sigmoid \space x = \frac{1}{1+e^{-x}}$
2. $tanh \space x = \frac{e^x-e^{-x}}{e^x+e^{-x}}$
3. $ReLU \space x = \max(0,x)$
4. $Leaky ReLU \space x=1(x<0)(ax)+1(x>=0)(x) $
5. Parametric  ReLU
6. Randomized Leaky ReLU



| 激活函数              | 优点                                               | 缺点                                                         |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| $sigmoid$             | 求导方便，$\sigma(x)’=\sigma(x)(1-\sigma(x))$      | 容易饱和而导致梯度消失；输出不是以0位中心的；幂运算相对耗时  |
| $tanh$                | 同上；缓解了梯度消失                               | 梯度消失；幂运算相对耗时                                     |
| ReLU                  | 收敛速度快；计算简单；解决了梯度爆炸和梯度消失问题 | “脆弱”（会导致神经元可能一直不会被激活）；输出不是以0位中心的； 很容易改变数据的分布 |
| Leaky ReLU            | 保留了$x<0$部分，不那么脆弱                        |                                                              |
| Parametric  ReLU      | 将$x<0$部分的斜率参数$\alpha$加入训练，效果更好    |                                                              |
| Randomized Leaky ReLU |                                                    |                                                              |
| Maxout                |                                                    |                                                              |

# 池化层

# 误差函数

问题：

1. CNN中误差函数为什么要用交叉熵？为什么不用MSE？
2. softmax和sigmoid有什么区别？

疑惑：

1. 常规理解里sigmoid函数的输入是一个值，根据$g(x)$是否大于阈值判断$\hat{y}$为1还是为0，是一个二分类问题。而常规理解中的softmax函数的输入是一个数组，$[1,2,3]$，使用softmax将这一数组变成一个概率分布$[g(1),g(2),g(3)]$。而在交叉熵中，softmax和sigmoid函数都是对数组$[y_1,y_2,...,y_i]$进行处理然后计算交叉熵$H(p,q)$。

   常说的“softmax适合于多分类问题”，指的是他可以将多个输出值变成一种概率分布，预测结果是每个分类的概率，对应的$label=[0,1,0]$，即$one \space hot$编码。如果对数组使用sigmoid函数，则应该是独立判断数组中的每一个值属于每一个类的概率，$label=[1,1,0]$。

## 交叉熵

> [softmax和交叉熵](https://zhuanlan.zhihu.com/p/27223959)

说交叉熵之前先介绍相对熵，相对熵又称为KL散度（Kullback-Leibler Divergence），用来衡量两个分布之间的距离， 记为$D_{KL}(p||q)$
$$
\begin{align}
D_{KL}(p||q) &=\sum_{x \in X}p(x)\log \frac{p(x)}{q(x)} \\
&= \sum_{x \in X}p(x)\log(p(x))-\sum_{x \in X}p(x)\log(q(x)) \\
&=-H(p)-\sum_{x \in X}p(x)\log(q(x)) \\
\end{align} \\
\\
$$
KL散度不具备对称性，即$D_{KL}(p||q)\neq D_{KL}(q||p)$其中$H(p)$是$p$的交叉熵。

假设有两个分布$p$和$q$，它们在给定样本集上的交叉熵定义为： 
$$
CE(p,q)=-\sum_{x \in X}p(x)\log(q(x))=H(p)+D_{KL}(p||q)
$$
从信息论的角度来说，使用q编码对p编码进行建模（用编码q来表示p），其所需信息量除了p本身的信息量之外，还需要平均的额外附加信息量$D_{KL}(p,q)$。很像一个条件概率，$H[p,q]=H[p]+H[q|p]$

从这里可以看出，交叉熵和相对熵相差了$H(p)$，而当$p$已知的时候，$H(p)$是个常数，所以交叉熵和相对熵在这里是等价的，反映了分布$p$和$q$之间的相似程度。 

> KL散度可以被用于计算代价，而在特定情况下（分布$p$已知时）最小化KL散度等价于最小化交叉熵，而交叉熵的运算更简单，所以用交叉熵当做代价。

以下摘自PRML

> 假设数据通过未知分布p(x)形成，我们想要对p(x)建模。我们可以试着使一些参数分布$q(x |θ)$来近似这个分布。$q(x|θ)$由可调节的参数θ控制（例如一个多元高斯分布）。一种确定θ的方式是最小化化$p(x)和q(x | θ)$之间关于θ的Kullback-Leibler散度。
> $$
> KL(p||q)=-\int p(x)\ln q(x)dx-(-\int p(x)\ln p(x)dx)
> $$
> 我们不能直接这么做，因为我们不知道p(x)。但是，假设我们已经观察到了服从分布p(x)的有限数量的训练点$x_n$，其 中n = 1, . . . , N。那么，关于p(x)的期望就可以通过这些点的有限加和，近似可得
> $$
> KL(p||q)=\frac{1}{N}( -\ln q(x|\theta)+\ln p(x))
> $$
> 第一项是使用训练集估计的分布$q(x | θ)下$的θ的负对数似然函数。因此我们看到，最小化Kullback-Leibler散度等价于最大化似然函数。  

对于一个多分类问题，假设其纬度为$C$，$p$一个$C$维向量，采用One-hot编码方式表示，如$[0,0,1,0]$，$t_i=1$表示当$i$为真实类别时，其概率为1。模型输出为$[\hat{y_1},\hat{y_2},...,\hat{y_C}]$，经过softmax之后，模型输出被转换成一个概率分布，记为$q$，则真实分布于模型预测分布的交叉熵可以表示为：
$$
l_{CE}=-\sum_{i=1}^Ct_i\log(y_i)
$$
而多分类问题的最大似然函数为：
$$
L(t|x)=\prod_{i=1}^CPr(t_i|x)=\prod_{i=1}^Cy_{i}^{t_i}
$$
取对数并加上负号：
$$
-\log p(t|x)=-\sum_{i=1}^Ct_i\log(y_i)
$$
即最大化似然函数就是最小化负对数似然函数，就是最小化分布期望分布p和训练分布数据q的交叉熵

> 对于基于高斯噪声的回归模型, 即Y=f(X)+ Z，Z 是高斯噪声，最大似然的目标函数一样reduce to cross entropy. 只不过在那种情形下，可以把目标函数进一步reduce, 最后就变成了square error. 



以下摘自Quora上[When should you use cross entropy loss and why?](https://www.quora.com/When-should-you-use-cross-entropy-loss-and-why)的回答

![1532997077374](pic/quora_answer_0.png)



通过以上，得到结论：“从概率角度来说，模型训练的目的就是找到一个分布q，与真实事件分布p一致，但是由于我们得不到真实事件的分布p，于是退而求其次，假设我们对真实事件进行了N次独立同分布的采样，得到分布p'，想通过训练模型使分布q接近于分布p'，从而近似接近分布p。衡量两个分布之间的距离使用KL散度，而在已知分布p'的情况下，最小化KL散度等价于最小化p'和q的交叉熵，等价于最大化似然概率函数。而在误差为高斯分布的假设下，最大化似然概率函数等价于最小化负对数似然概率，等价于最小化训练数据与期望数据的均方误差。”

## TF中不同交叉熵函数的对比

摘自知乎专栏[TensorFlow中不同交叉熵函数对比与总结](https://zhuanlan.zhihu.com/p/39374523)

主要说明两个函数的不同

1. tf.nn.sigmoid_cross_entropy_with_logits
2. tf.nn.softmax_cross_entropy_with_logits

> 首先，Sigmoid其实是Softmax的一个特例，Softmax用于多分类问题，而Sigmoid用于二分类问题。但是TensorFlow中这个函数更加泛化，允许在类别彼此独立或者不互斥的情况下处理多标签分类问题，换言之就是可以**一次性处理N个二分类问题**。比如我们的模型判断一张图片中是否同时包含大象和狗，label值可以取[0,0]，[0,1]，[1,0]，[1,1]。

对于验证码识别问题，验证码识别是识别[a-zA-Z0-9]共62个字符（少位按“_”补齐），在不顺序识别的情况下，模型可以使用63个二分类问题对验证码进行识别，为了保证顺序，则label纬度为$1\times 62*4$，4为验证码个数，每一个63使用一个One hot编码。

tf.nn.sigmoid_cross_entropy_with_logits适用于一次性处理N个二分类问题，即识别图中是否有多个类，属于独立不互斥任务。

tf.nn.softmax_cross_entropy_with_logits用于多项互斥分类任务。例如CIFAR-10中图片只能分一类而不像前面判断是否包含多类动物。其label需要使用One hot编码，输入的数据logits不能自行使用softmax，函数内部会实现softmax处理。

另外有两篇文章也说了这两个函数的区别

> [[Tensorflow sigmoid and cross entropy vs sigmoid_cross_entropy_with_logits](https://stackoverflow.com/questions/46291253/tensorflow-sigmoid-and-cross-entropy-vs-sigmoid-cross-entropy-with-logits)]
>
> [[How to choose cross-entropy loss in tensorflow?](https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow)]

总结一下，Multi-class cross-entropy中，计算的是`tf.nn.softmax_cross_entropy_with_logits` ：

```python
-tf.reduce_sum(p * tf.log(q), axis=1)
```

这里的p和q分别是label和训练输入数据的**概率分布**

而在Binary cross-entropy中，计算公式为：

```python
p * -tf.log(q) + (1 - p) * -tf.log(1 - q)
```

这里的p和q都是一个数，是class A的概率（其实就是二项分布）。如果p是一个向量，则向量的每一个元素对应的是独立的二分类问题。

> If `p` is a vector, each individual component is considered an *independent binary classification*. See [this answer](https://stackoverflow.com/a/47034889/712995) that outlines the difference between softmax and sigmoid functions in tensorflow. So the definition `p = [0, 0, 0, 1, 0]` doesn't mean a one-hot vector, but 5 different features, 4 of which are off and 1 is on. The definition `q = [0.2, 0.2, 0.2, 0.2, 0.2]` means that each of 5 features is on with 20% probability.   

因此`tf.nn.sigmoid_cross_entropy_with_logits`对应的lable并不是一个one hot编码，而是多个独立的二分类问题。

相反`tf.nn.softmax_cross_entropy_with_logits`对应的label是一个one hot编码，

### 拓展

One-hot-encode 存在几个问题：

1. 不能表示顺序
2. 当类别超多时，会浪费空间，甚至导致内存溢出

问题1就是验证码识别的问题，验证码识别是一个不互斥多分类问题，且必须保证识别结果的顺序，所以采用一个$1\times num\_classes$的label肯定是不行的，因此采用了一个$1\times num\_classes*num\_captcha$的label来训练。

对于问题2，TF中提供了[Sparse Function family](https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow#47034889)，不采用OHE对label进行编码，该函数的labels不是one-hot向量，而是每个类别的**index**。 

> ## Sparse functions family
>
> - [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)
> - [`tf.losses.sparse_softmax_cross_entropy`](https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy)
> - [`tf.contrib.losses.sparse_softmax_cross_entropy`](https://www.tensorflow.org/api_docs/python/tf/contrib/losses/sparse_softmax_cross_entropy) (DEPRECATED)
>
> Like ordinary `softmax` above, these loss functions should be used for multinomial mutually exclusive classification, i.e. pick one out of `N` classes. The difference is in labels encoding: the classes are specified as integers (class index), not one-hot vectors. Obviously, this doesn't allow soft classes, but it can save some memory when there are thousands or millions of classes. However, note that `logits` argument must still contain logits per each class, thus it consumes at least `[batch_size, classes]` memory.
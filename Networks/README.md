# Networks

## Fundamental

### CNNs

- **LeNet (1995)**: [**Paper**](https://ieeexplore.ieee.org/document/726791) | [**Code**]( cnns/le_net.py ) | [**Doc**](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)
- **AlexNet (2012)**: [**Paper**](https://dl.acm.org/doi/10.1145/3065386) | [**Code**]( cnns/alex_net.py) | [**Doc**](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
- **VGG (2014)**: [**Paper**](https://arxiv.org/abs/1409.1556) | [**Code**]( cnns/vgg.py ) | [**Doc**](https://d2l.ai/chapter_convolutional-modern/vgg.html)
- **GoogLeNet (2015)**: [**Paper**](https://ieeexplore.ieee.org/document/7298594) | [**Code**]( cnns/google_net.py ) | [**Doc**](https://d2l.ai/chapter_convolutional-modern/googlenet.html)
- **ResNet (2016)**: [**Paper**](https://ieeexplore.ieee.org/document/7780459) | [**Code**]( cnns/res_net.py) | [**Doc**](https://d2l.ai/chapter_convolutional-modern/resnet.html)



基本的 AI 任务：回归、分类、聚类

- [**GAN**](https://juejin.cn/post/7120443914854613029) [**Code**](https://github.com/wsj-create/GAN/blob/master/test_gan.py)



## Finance Transformer (FiT)

> A new **Plug-and-Play module** for **Finance Cross-Section Feature** Extraction.

### What is Cross-Section Feature ?

Cross-Section feature refers to the observed values of the **same feature of a group of samples** reflecting an aggregate at the same time cross-section and is one of the common types of sample data.

Cross-section data are generally characterized by the following:
- It is characterized at one point in time, with **no temporal information**.
- The order of the samples on the cross-section is changeable, i.e., the **sample order contains no information**.
- Some samples **may not contain information**.

<img src="./README.assets/What is Cross-Section Feature.png" alt="What is Cross-Section Feature.png" style="zoom:60%;" />

### Why we need FiT ?

Since the cross-section data has several features as described above, we want to have an information extraction module that satisfies the following properties:
- The extracted information of the module is **not affected by the order of the samples**.
- Samples without information **can be ignored**.

These two properties are at the heart of why Finance Transformer was proposed !

### What is the FiT ?

The **framework** of FiT is shown below.

<img src="./README.assets/FiT_Framework.png" alt="FiT_Framework.png" style="zoom:50%;" />

FiT achieves the top 2 goals through a **two-part design**, respectively:
- Similar to the ViT architecture, the `[fin]` token is introduced to **summarize information**. However, since there is **no positional encoding**, it can be realized that the **output is independent of the order of the inputs**.
- Perform **Attention with Mask** to ignore the parts that do not contain information.

<img src="./README.assets/detail_of_FiT.png" alt="detail_of_FiT.png" style="zoom:50%;" />

A **demo** using FiT:

<img src="./README.assets/FiT_Demo.png" alt="FiT_Demo.png" style="zoom:50%;" />

### How to use FiT ?

```python
# random the input
features = torch.randn(64, 5, 768, dtype=torch.float32)
masks = torch.randint(low=0, high=2, size=(64, 5), dtype=torch.long)

# init the model
fit = FiT(
    input_size=768,
    dim=64,
    depth=1,
    heads=2,
    ff_hidden_dim=32
)

# forward
output = fit(features, masks)
print(output)
```


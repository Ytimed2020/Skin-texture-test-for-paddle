

# 基于飞桨框架的痤疮医学检测系统

通过Resnet50模型进行训练实现，进行常见皮肤医学影像痤疮进行检测，并且最终通过paddlelite进行模型部署，最终可达到85%的精度！

## 一、项目背景

痤疮也称为[青春痘](https://baike.baidu.com/item/青春痘/679380)，面部痤疮就是脸上长青春痘，这是一种常见的皮肤病，一般多见于青少年，当然面部[痤疮](https://baike.baidu.com/item/痤疮)也没有年龄限制。面部痤疮有很多种类型，面部粉刺、丘疹、脓疱、结节等多形性皮损为主。严重的还会化脓，因此面部痤疮需要及时进行[治疗](https://baike.baidu.com/item/治疗/8948764)。需要特别注意的是，护肤品对痤疮只是辅助护理，痤疮作为一种常见皮肤病应该去正规医院进行治疗，避免去一些不良美容院浪费不必要的钱财和时间二，而通过我们训练，会协助检测面部青春痘，并在未来加入护肤品推荐以及美容院推荐系统，做一个检验全流程项目。

## 二、数据集来源

数据集来源于爬虫以及国外医学网站获取，其中痤疮分多类痤疮，并进行了数据手动筛除。

## 1.数据加载和预处理

```python
import paddle
import warnings
import os
import random
import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
from paddle.onnx import export
from paddlex.cls import transforms
import cv2
import numpy as np
import paddlex as pdx
import imgaug.augmenters as iaa
from tqdm import tqdm
import chardet
```

```python
  # 读取一张图片进行测试
  lena = mpimg.imread('laopo/1.jpg')  
  #读取和代码处于同一目录下的 lena.png
   print(lena.shape)
   plt.imshow(lena)  # 显示图片
   plt.axis('off')  # 不显示坐标轴
   plt.show()
```

```python
dirpath = "chuochuang"

def get_all_txt():
    all_list = []
    i = 0
    for root, dirs, files in os.walk(dirpath): 
        for file in files:
            i = i + 1
            if ("chuochuang" in root):
                all_list.append(os.path.join(root, file) + " 0\n")
    allstr = ''.join(all_list)
    f = open('list.txt', 'w', encoding='utf-8')
    f.write(allstr)
    return all_list, i

all_list, all_lenth = get_all_txt()
random.shuffle(all_list)
random.shuffle(all_list)
```

```python
train_size = int(all_lenth * 0.82)
train_list = all_list[:train_size]
val_list = all_list[train_size:]
```

```python
train_txt = ''.join(train_list)
f_train = open('train\train_list.txt', 'w', encoding='utf-8')
f_train.write(train_txt)
f_train.close()

#生成txt
val_txt = ''.join(val_list)
f_val = open('val/val_list.txt', 'w', encoding='utf-8')
f_val.write(val_txt)
f_val.close()
```

# 三、模型选择和开发

本次训练中选取了resnet50模型

### Resnet的模型结构

当我们对输入做了卷积操作后会包含4个残差, 最后进行全连接操作以便于进行分类任务，Resnet50相当于包含50个conv2d操作。

![image-20210815145021518](C:\Users\杨毓栋\AppData\Roaming\Typora\typora-user-images\image-20210815145021518.png)

我们都知道堆叠网络难以收敛，梯度消失在一开始就阻碍网络的收敛。所以未解决以上问题，残差结构题出residual层，明确的让这些层拟合残差映射，将期望的基础映射表示为H(x)，我们将堆叠的非线性层拟合另一个映射F(x)=H(x)−x。原始的映射重写为F(x)+x。我们假设残差映射比原始的、未参考的映射更容易优化。在极端情况下，如果一个恒等映射是最优的，那么将残差置为零比通过一堆非线性层来拟合恒等映射更容易。快捷连接简单地执行恒等映射，并将其输出添加到堆叠层的输出。恒等快捷连接既不增加额外的参数也不增加计算复杂度。整个网络仍然可以由带有反向传播的SGD进行端到端的训练。

而  ResNet（Residual Neural Network）由微软研究院的Kaiming He等四名华人提出，通过使用ResNet Unit成功训练出了152层的神经网络，并在ILSVRC2015比赛中取得冠军，在top5上的错误率为3.57%，同时参数量比VGGNet低，效果非常突出。ResNet的结构可以极快的加速神经网络的训练，模型的准确率也有比较大的提升。同时ResNet的推广性非常好，甚至可以直接用到InceptionNet网络中。

ResNet的主要思想是在网络中增加了直连通道，即Highway Network的思想。所以ResNet的思想允许原始输入信息直接传到后面的层中。这样的话这一层的神经网络可以不用学习整个的输出，而是学习上一个网络输出的残差，因此ResNet又叫做残差网络。

![image-20210815145410977](C:\Users\杨毓栋\AppData\Roaming\Typora\typora-user-images\image-20210815145410977.png)

- **因resnet50网络层数更多，训练时间更长，因此考虑在GPU上进行训练**



```python
train_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotate(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(224),
    transforms.Normalize()
])


train_dataset = pdx.datasets.ImageNet(
    data_dir=train_base,
    file_list='train_list.txt',
    label_list='labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir=eval_base,
    file_list='eval_list.txt',
    label_list='labels.txt',
    transforms=eval_transforms,
    shuffle=True)


num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet50_vd(num_classes=num_classes)
model.train(num_epochs=320,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            save_interval_epochs=20,
            train_batch_size=32,
            lr_decay_epochs=[4, 6, 8],
            learning_rate=2e-4,
            save_dir='output/ResNet50_vd',
            log_interval_steps=50,
            use_vdl=True)
            
```

# 模型评估

```python
model = pdx.load_model('ResNet50_vd/best_model')
model.evaluate(eval_dataset, batch_size=16, epoch_id=None, return_details=False)
```



# Android端部署

### 转换并生成.nb文件

```python
!paddlex --export_inference --model_dir=output/ResNet50_vd/best_model --save_dir=./inference_model --fixed_input_shape=[224,224]
```

```python
!paddle_lite_opt \
    --model_file=inference_model/__model__ \
    --param_file=inference_model/__params__ \
    --optimize_out_type=naive_buffer \
    --optimize_out=cpu_model \
    --valid_targets=arm
```

# 五、总结与升华

​      通过学习大佬们的开源项目以及熟悉深度学习框架，整体属于一个小白的尝试，也是刚接触不久飞桨社区的尝试，在未来中，我将会更多的写出复杂的深度学习项目，并且将该项目部署基于flask部署到web端上做一个青春痘助诊项目，希望大家可以多多支持。

# 六、个人介绍

​			中北大学大一在读，热衷兴趣方向为CV，刚接触深度学习不久，属于小白一只，从开发屁巅屁颠跑向深度学习岗。

​			中国人工智能学会CAAI会员、百度高校领航团团长。

​			个人及团队获纳安集团战略合作伙伴，中国高校计算机大赛西北赛区二等奖、中国大学生计算机设计大赛二等奖、大学生创新创业大赛立项、互联网+校级奖项*2项、软件著作权*4份、蓝桥杯C/C++省级奖项等多项比赛奖项（均为第一负责人）
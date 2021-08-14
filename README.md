# 一、手把手教你复现ResNet50
* aistudio地址：[https://aistudio.baidu.com/aistudio/projectdetail/2282003](https://aistudio.baidu.com/aistudio/projectdetail/2282003)
* github地址： [https://github.com/livingbody/resnet-livingbody.git](https://github.com/livingbody/resnet-livingbody.git)

>众所周知Tomato Donkey老师已经复现了AlexNet给大家打了样，手把手教了如何复现，今天周末就跟着跑一趟把ResNet复现了

![](https://ai-studio-static-online.cdn.bcebos.com/c379c1c587db4c8b8cd50823e8bcd68432cc5d2efa5f4dedb8fe831b69f916cf)

课程地址： [https://aistudio.baidu.com/aistudio/education/group/info/24681](https://aistudio.baidu.com/aistudio/education/group/info/24681)

# 二、环境设置

```
pip install torch
pip install torchvision
pip install torchsummary
pip install paddlepaddle
```

# 三、ResNet代码转换


![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fdeveloper.ibm.com%2Fdeveloper%2Farticles%2Fusing-deep-learning-to-take-on-covid-19%2Fimages%2FResNet50-1.png&refer=http%3A%2F%2Fdeveloper.ibm.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1631503999&t=d820c03c37b4901e5ab12293db0f122e)


>以torchvision\models\resnet.py为蓝本开始修改


## 1. nn等替换
>不可避免需要替换网络的重要部件，这里简要列举我改的部分

![](https://ai-studio-static-online.cdn.bcebos.com/145b92ce4f4d4ab680be5917281b258b2776afe1334845139f39f0c533d979db)

>除此之外，还需要注意:
```
nn.module  -->  nn.Layer
class ResNet(nn.Module): --> class ResNet:(nn.Layer):
```
>conv2d替换
```
torch:
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

paddle:                     
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2D:
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)   
```                  
>relu替换

`self.relu = nn.ReLU(inplace=True)  --> self.relu = nn.ReLU()`

##  2.预训练模型加载
>pytorch中通过 load_state_dict_from_url来加载，此处需要改为PaddlePaddle加载方式，对比如下：

>torch改代码：

```
def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

>paddlepaddle方式

```
# added by ken
def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        load_dygraph_pretrain(model, pretrained)
    return model
```


##  3. 权重初始化
> **nn.init.* 对应 paddle.nn.initializer.*** ，但是初始化方法不一样，一个是直接赋值，一个是生成，再赋值

> __init__方法中权重初始化torch代码
 
```
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
        
        

if zero_init_residual:
    for m in self.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
```

> __init__方法中权重初始化paddlepaddle代码

```

# for m in self.modules():
for m in self.sublayers():
    print(m)
    if isinstance(m, nn.Conv2D):
        # 该接口实现Kaiming正态分布方式的权重初始化
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        print(50 * '*')
        m.weight_attr = init_weights(init_type='kaiming')
        m.bias_attr = init_weights(init_type='kaiming')
    elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
        # nn.initializer.Constant(m.weight, 1)
        # m.weight = nn.initializer.Constant(1)
        # nn.initializer.Constant(m.bias, 0)
        # m.bias = nn.initializer.Constant(0)
        m.param_attr = init_weights(init_type='constant')
        m.bias_attr = init_weights(init_type='constant')
        
        
if zero_init_residual:
    for m in self.modules():
        if isinstance(m, Bottleneck):
            # m.bn3.weight = nn.initializer.Constant(0)  # type: ignore[arg-type]
            m.bn3.weight = init_weights(init_type='constant')  # type: ignore[arg-type]
        elif isinstance(m, BasicBlock):
            # m.bn2.weight = nn.initializer.Constant(0)  # type: ignore[arg-type]
            m.bn2.weight = init_weights(init_type='constant')  # type: ignore[arg-type]

```


>init_weights方法定义

```
import paddle


def init_weights(init_type='constant'):
    if init_type == 'constant':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant())
    if init_type == 'normal':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal())
    elif init_type == 'xavier':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
    elif init_type == 'kaiming':
        return paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

```


# 四、ResNet模型权重转换

## 1.引入必要的包
```
import numpy as np
import torch
import ResNet_paddle.paddlevision.models
from torchsummary import summary
from torchvision.models import resnet50
import paddle
from collections import OrderedDict

```

## 2.查看pytorch权重文件信息
```
def model_summary():
    model = resnet50()
    checkpoint = torch.load("resnet50-0676ba61.pth")
    model.load_state_dict(checkpoint)
    summary(model, (3, 224, 224))
    for name in model.state_dict():
        print(name)

```

## 3.查看pytorch网络各层名称

```
def show_layer_name():
    model = resnet50()
    for name in model.state_dict():
        print(name)
```
## 4.保存pytorch网络权重名称
```
def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')
```

## 5.pytorch权重转paddlepaddle权重
```
def transfer():
    res2net_paddle_implement = paddle.vision.models.resnet50(pretrained=False)
    export_weight_names(res2net_paddle_implement)  # 将自己paddle模型的keys存为txt
    paddle_list = open('paddle.txt')  # paddle的keys
    state_dict = torch.load('resnet50-0676ba61.pth')

    paddle_state_dict = OrderedDict()
    paddle_list = paddle_list.readlines()
    torch_list = state_dict.keys()
    for p in paddle_list:
        p = p.strip()
        t = p
        if "mean" in p:
            t = p.replace("_mean", "running_mean")
        if "variance" in p:
            t = p.replace("_variance", "running_var")
        if t in torch_list:
            if 'fc' not in p:
                paddle_state_dict[p] = state_dict[t].detach().cpu().numpy()
            else:
                paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T
        else:
            print(p)

    f = open('resnet50.pdparams', 'wb')
    import pickle
    pickle.dump(paddle_state_dict, f)
    f.close()

```
## 6. 模型文件替换
>model_urls修改，用上面方法生成的pdparams文件替换，用到 **resnet50** 举例

```
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet50': r'H:\论文复现0814\ResNet50-livingbody\resnet50.pdparams',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
```
>只使用其中的resnet50，修改resnet50为转换后的 **resnet50.pdparams** ，其他以此类推。

## 7.测试
```
def test_forward():
    model_torch = resnet50()
    model_paddle = ResNet_paddle.paddlevision.models.resnet50()
    model_torch.eval()
    model_paddle.eval()
    torch_checkpoint = torch.load('resnet50-0676ba61.pth')
    # model_torch.load_state_dict(torch_checkpoint['model'])
    model_torch.load_state_dict(torch_checkpoint)
    paddle_checkpoint = paddle.load('resnet50.pdparams')
    model_paddle.set_state_dict(paddle_checkpoint)

    x = np.random.randn(1, 3, 224, 224)
    input_torch = torch.tensor(x, dtype=torch.float32)
    out_torch = model_torch(input_torch)

    input_paddle = paddle.to_tensor(x, dtype='float32')
    out_paddle = model_paddle(input_paddle)
    # 查看torch权重
    print('torch result:{}'.format(out_torch))
    print('torch shape result:{}'.format(out_torch.shape))
    # 查看paddlepaddle权重
    print('paddlepaddle result:{}'.format(out_paddle))
    print('paddlepaddle shape result:{}'.format(out_paddle.shape))

    print(50*'*')
    # 查看权重差值
    out_diff=out_torch.detach().numpy()-out_paddle.numpy()
    print("diff:", np.max(np.abs(out_diff)))
    assert np.allclose(out_torch.detach(), out_paddle, atol=1e-2)
```

>最终pytorch与paddlepaddle权重差最大为
```
**************************************************
diff: 2.6226044e-06
```


# 五、注意事项

特别要注意的是虽然有些方法名称是一样的，但是部分参数名称有变化，例如PaddlePaddle中的Conv2D中的bias_attr对应torch中的bias，PaddlePaddle中的ReLU对应torch中的ReLU(inplace=True)，inplace啥意思我也不知道，有人说这个会加速运算！


```python

```

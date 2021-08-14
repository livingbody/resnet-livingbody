import numpy as np
import torch
from torchvision import models
import ResNet_paddle.paddlevision.models
from torchsummary import summary
from torchvision.models import resnet50
import paddle
from collections import OrderedDict


# 查看pytorch权重文件信息
def model_summary():
    model = resnet50()
    checkpoint = torch.load("resnet50-0676ba61.pth")
    model.load_state_dict(checkpoint)
    summary(model, (3, 224, 224))
    for name in model.state_dict():
        print(name)


def show_layer_name():
    model = resnet50()
    for name in model.state_dict():
        print(name)

def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')


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


if __name__ == '__main__':
    # model_summary()
    # transfer()
    test_forward()
    # show_layer_name()

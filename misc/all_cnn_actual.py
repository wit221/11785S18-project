from torch.nn import Module, Sequential, Conv2d, ReLU, Dropout, Dropout2d, AvgPool2d, Linear, BatchNorm2d, Softmax

class Flatten(Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """

    #DROPOUT_INPUT = 0.2
    #DROPOUT_OTHER = 0.5
    #DROPOUT_INPUT = 0.025
    #DROPOUT_OTHER = 0.025
    DROPOUT_INPUT = 0.1
    DROPOUT_OTHER = 0.2


    layers = []
    if True:
        layers += [Dropout(DROPOUT_INPUT)]
        layers += [Conv2d(3, 96, 3, stride=1, padding=1), ReLU(inplace=True)]
        layers += [Conv2d(96, 96, 3, stride=1, padding=1), ReLU(inplace=True)]
        layers += [Conv2d(96, 96, 3, stride=2, padding=1), ReLU(inplace=True)]
        #layers += [BatchNorm2d(96)]
        # 16x16 96 channels
        layers += [Dropout(DROPOUT_OTHER)]
        layers += [Conv2d(96, 192, 3, stride=1, padding=1), ReLU(inplace=True)]
        # 16x16 192 channels
        layers += [Conv2d(192, 192, 3, stride=1, padding=1), ReLU(inplace=True)]
        # 16x16 192 channels
        layers += [Conv2d(192, 192, 3, stride=2, padding=1), ReLU(inplace=True)]
        #layers += [BatchNorm2d(192)]
        # 8x8 192 channels
        layers += [Dropout(DROPOUT_OTHER)]
        layers += [Conv2d(192, 192, 3, stride=1, padding=0), ReLU(inplace=True)]
        # 6x6 192 channels
        layers += [Conv2d(192, 192, 1, stride=1, padding=0), ReLU(inplace=True)]
        # 6x6 192 channels
        layers += [Conv2d(192, 10, 1, stride=1, padding=0), ReLU(inplace=True)]
        #layers += [BatchNorm2d(10)]
        # 6x6 10 channels
        layers += [AvgPool2d(6)]
        # 1x1 10 channels
        layers += [Flatten()]
    else:
        layers += [ Conv2d(3, 1, kernel_size=3, padding=1) , ReLU(inplace = True) ]
        layers += [ Flatten() ]
        #layers += [ Linear(1024, 100), ReLU(inplace = True) ]
        #layers += [ Linear(100, 10), ReLU(inplace = True) ]
        layers += [ Linear(1024, 1000), ReLU(inplace = True) ]

    return Sequential(*layers)

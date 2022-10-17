import moai.nn.convolution as mic
import moai.nn.activation as mia

import torch

__all__ = [
    "Bottleneck",
    "PreResBottleneck",
    "PreActivBottleneck",
    "LambdaBottleneck"
]

'''
    Bottleneck versions with 3 convolutions (2 projections, 1 bottleneck)    
'''
class Bottleneck(torch.nn.Module):
    def __init__(self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(Bottleneck, self).__init__()
        self.W1 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            **convolution_params
        )
        self.A1 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            **convolution_params
        )
        self.A2 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )
        self.W3 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=out_features,          
            **convolution_params
        )
        self.A3 = mia.make_activation(
            features=out_features,
            activation_type=activation_type,
            **activation_params
        )
        self.S = torch.nn.Identity() if in_features == out_features\
            else mic.make_conv_1x1(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    **convolution_params,
                # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                ) if not strided else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                    **convolution_params,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(self.A2(self.W2(self.A1(self.W1(x)))))  # y = W3 * A2(W2 * A1(W1 * x))
        return self.A3(self.S(x) + y)                       # out = A3(S(x) + y)

class PreResBottleneck(Bottleneck):
    def __init__(self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(PreResBottleneck, self).__init__(
            convolution_type=convolution_type,
            activation_type=activation_type,
            in_features=in_features,
            out_features=out_features,
            bottleneck_features=bottleneck_features,
            convolution_params=convolution_params,
            activation_params=activation_params,
            strided=strided
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.A3(self.W3(self.A2(self.W2(self.A1(self.W1(x))))))  # y = A3(W3 * A2(W2 * A1(W1 * x)))
        return self.S(x) + y                                         # out = S(x) + y

class PreActivBottleneck(torch.nn.Module):
    def __init__(self,
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool,
    ):
        super(PreActivBottleneck, self).__init__()
        self.A1 = mia.make_activation(
            features=in_features,
            activation_type=activation_type,
            **activation_params
        )
        self.W1 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            **convolution_params
        )
        self.A2 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )
        self.W2 = mic.make_conv_3x3(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=bottleneck_features,
            **convolution_params
        )
        self.A3 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )
        self.W3 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=out_features,          
            **convolution_params
        )
        self.S = torch.nn.Identity() if in_features == out_features\
            else mic.make_conv_1x1(
                convolution_type=convolution_type,
                in_channels=in_features,
                out_channels=out_features,
                **convolution_params,
            # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
            ) if not strided else mic.make_conv_3x3(
                convolution_type=convolution_type,
                in_channels=in_features,
                out_channels=out_features,
                stride=2,
                **convolution_params,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W3(self.A3(self.W2(self.A2(self.W1(self.A1(x)))))) # y = W3 * A3(W2 * A2(W1 * A1(x)))
        return self.S(x) + y                                        # out = x + y



class LambdaConv(torch.nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        heads=4, 
        k=16, 
        u=1, 
        m=23
    ):
        super(LambdaConv, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(k * heads)
        )
        self.keys = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = torch.nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = torch.nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = torch.nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = torch.nn.functional.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class LambdaBottleneck(torch.nn.Module):
    def __init__(self, 
        convolution_type: str,
        activation_type: str,
        in_features: int,
        out_features: int,
        bottleneck_features: int,
        convolution_params: dict,
        activation_params: dict,
        strided: bool):
        super(LambdaBottleneck, self).__init__()
        #self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) #W1
        #self.bn1 = torch.nn.BatchNorm2d(planes) #A1
        self.W1 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=in_features,
            out_channels=bottleneck_features,
            stride=2 if strided else 1,
            **convolution_params
        )
        self.A1 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )

        # self.conv2 = torch.nn.ModuleList([LambdaConv(planes, planes)])
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.conv2.append(torch.nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        # self.conv2.append(torch.nn.BatchNorm2d(planes))
        # self.conv2.append(torch.nn.ReLU())
        # self.conv2 = torch.nn.Sequential(*self.conv2)

        self.W2 = LambdaConv(
            bottleneck_features,
            bottleneck_features)
        
        self.A2 = mia.make_activation(
            features=bottleneck_features,
            activation_type=activation_type,
            **activation_params
        )

        self.W3 = mic.make_conv_1x1(
            convolution_type=convolution_type,
            in_channels=bottleneck_features,
            out_channels=out_features,          
            **convolution_params
        )
        self.A3 = mia.make_activation(
            features=out_features,
            activation_type=activation_type,
            **activation_params
        )
        self.S = torch.nn.Identity() if in_features == out_features\
            else mic.make_conv_1x1(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    **convolution_params,
                # using a 3x3 conv for shortcut downscaling instead of a 1x1 (used in detectron2 for example)
                ) if not strided else mic.make_conv_3x3(
                    convolution_type=convolution_type,
                    in_channels=in_features,
                    out_channels=out_features,
                    stride=2,
                    **convolution_params,
                )

        # self.conv3 = torch.nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # self.bn3 = torch.nn.BatchNorm2d(self.expansion * planes)

        # self.shortcut = torch.nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = torch.nn.Sequential(
        #         torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
        #         torch.nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        # out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        # out = self.conv2(out)
        # out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        # out = torch.nn.functional.relu(out)
        # return out
        y = self.W3(self.A2(self.W2(self.A1(self.W1(x)))))
        return self.A3(self.S(x) + y)
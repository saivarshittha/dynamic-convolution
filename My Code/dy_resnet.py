import torch
import torch.nn as nn
from dy_conv import Dynamic_conv2d   

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

def conv1x1(in_planes,out_planes,stride = 1): 
    return Dynamic_conv2d(in_planes,out_planes,kernel_size = 1,stride = stride,bias = False,)   

def conv3x3(in_planes,out_planes,stride = 1,groups = 1,dilation = 1): # conv3x3 for dynamic convolution
    return Dynamic_conv2d(in_planes,out_planes,kernel_size = 3,stride = stride,padding = dilation,groups = groups,bias = False,dilation = dilation)

class BasicBlock(nn.Module): # expansion = 1, dilation = 1 , base_width = 64 ,groups = 1
    expansion = 1
    
    def __init__(self,in_planes,out_planes,stride = 1,downsample = None,
                 groups = 1,base_width = 64,dilation = 1,norm_layer = None):
        
        super(BasicBlock,self).__init__()
        
        if base_width != 64:
            raise ValueError('BasicBlock supports only base_width = 64')
        if groups != 1: 
            raise ValueError('BasicBlock supports only groups = 1')
        if dilation > 1:
            raise NotImplementedError('BasicBlock doesnot support dilation > 1')
        # self.conv1 and self.downsample layers downsample the input when stride != 1
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(in_planes,out_planes,stride)
        self.bn1   = norm_layer(out_planes)
        self.relu  = nn.ReLU() # modify input directly.
        self.conv2 = conv3x3(out_planes,out_planes)
        self.bn2   = norm_layer(out_planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,z):
        
        identity = z
        
        out = self.conv1(z)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(z)
        
        out = out + identity
        out = self.relu(out)
        
        return out
        
        
class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self,in_planes,out_planes,stride = 1,downsample = None,
                groups = 1,base_width = 64,dilation = 1,norm_layer = None):
        super(Bottleneck,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(out_planes *(base_width/64.)) * groups  ## ?? 
        
        self.conv1  = conv1x1(in_planes,width)
        self.bn1    = norm_layer(width)
        self.conv2  = conv3x3(width,width,stride,groups,dilation)
        self.bn2    = norm_layer(width)
        self.conv3  = conv1x1(width,out_planes * self.expansion)
        self.bn3    = norm_layer(out_planes * self.expansion)
        self.relu   = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,z):
        identity    = z
        
        out         = self.conv1(z)
        out         = self.bn1(out)
        out         = self.relu(out)
        
        out         = self.conv2(out)
        out         = self.bn2(out)
        out         = self.relu(out)
        
        out         = self.conv3(out)
        out         = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(z)
        
        out = out + identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    
    def __init__(self,block,layers,num_classes = 1000,zero_init_residual = False,
                groups = 1,width_per_group = 64,replace_stride_with_dilation = None,
                norm_layer = None):
        
        super(ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        print('XXX')
        self.in_planes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace 
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False,False,False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("Invalid argument : Size error for replace_stride_with_dilation")
        
        self.groups        = groups
        self.base_width    = width_per_group
        
        self.conv1         = nn.Conv2d(3,self.in_planes,kernel_size = 7,stride = 2,padding = 3,
                                      bias = False)
        self.bn1           = norm_layer(self.in_planes)
        self.relu          = nn.ReLU()
        self.maxpool       = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
        self.layer1        = self._make_layer(block,64,layers[0])
        self.layer2        = self._make_layer(block,128,layers[1],stride = 2,
                                             dilate = replace_stride_with_dilation[0])
        self.layer3        = self._make_layer(block,256,layers[2],stride = 2,
                                             dilate = replace_stride_with_dilation[1])
        self.layer4        = self._make_layer(block,512,layers[3],stride = 2,
                                              dilate = replace_stride_with_dilation[2])
        self.avgpool       = nn.AdaptiveAvgPool2d((1,1))
        self.fc            = nn.Linear(512 * block.expansion,num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity = 'relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m,BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)
        
    def update_temperature(self):
        for m in self.modules():
            if isinstance(m,Dynamic_conv2d):
                m.update_temperature()              ### ???
    
    def _make_layer(self,block,out_planes,blocks,stride = 1,dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                            conv1x1(self.in_planes,out_planes * block.expansion,stride),
                            norm_layer(out_planes * block.expansion),
                            )
        layers = []
        layers.append(block(self.in_planes,out_planes,stride,downsample,self.groups,
                           self.base_width,previous_dilation,norm_layer))
        self.in_planes = out_planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_planes,out_planes,groups = self.groups,
                               base_width = self.base_width,dilation = self.dilation,
                               norm_layer = norm_layer))
        return nn.Sequential(*layers)
    
    def _forward_impl(self,z):
        print('1')
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)
        print('2')
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        
        z = self.avgpool(z)
        z = torch.flatten(z,1)
        z = self.fc(z)
        print('3')
        
        return z
    def forward(self,z):
        return self._forward_impl(z)
    
      
        
        
def _resnet(arch,block,layers,pretrained,progress, **kwargs):
    model = ResNet(block,layers, **kwargs)
    print("Hi")
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress = progress)
        model.load_state_dict(state_dict)  
    print('This is _resnet func')     
    return model
def resnet18(pretrained = False,progress = True,**kwargs):
    print("hey")
    return _resnet('resnet18',BasicBlock,[2,2,2,2],pretrained,progress,**kwargs)
  

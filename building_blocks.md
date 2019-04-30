This is a summary of commonly used 'building blocks' for your PyTorch projects which have been gathered from different sources over the last year. Use them carefully, at best have a look at the original paper to learn more about them. Most of them have been used and tested on PyTorch 1.0 and worked well. Usually, there are other implementations available. The ones you find here were just the most convenient ones to use on a personal level.

**Disclaimer:**
*I added sources on how I found the implementations. Those might be unofficial sources or not the first publications using the mentioned methods. If you find any error let me know and I will update this summary.*


### Spectral Normalization

|[Github Source](https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py)|[Original Paper](https://arxiv.org/abs/1802.05957)|
|-|-|

##### Usage
Wrap the layer to which you want to apply spectral normalization.

E.g. 
```python
class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Number of feature in first layer
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.main(x)
```

With spectral normalization:

```python
class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Number of feature in first layer
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.main(x)
```

##### Code

```python
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
```


### Adaptive Instance Normalization

|[Github Source](https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py)|
|-|

##### Usage

We can for example change resnet blocks to use adaptive instance normlization the following way.

Note that the code here doesn't represent a good way of building a resnet block. It's just a simplified example to illustrate the process.
```python
class ResnetBlock(nn.Module):
    """Define a resnet block"""

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        
        self.conv_block(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
```

With adaptive instance norm:

```python
class ResnetBlockX(nn.Module):
    """Define a resnet block with adaptive instance norm"""

    def __init__(self, dim, dim_style):
        super(ResnetBlockX, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.adain1 = AdaptiveInstanceNorm(dim, dim_style)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.adain2 = AdaptiveInstanceNorm(dim, dim_style)

    def forward(self, x, style):
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)
        out = self.conv2(out)

        out = x + self.adain2(out, style)
        return out
```

Don't forget that we now have two inputs to the forward pass (x, style). Building a small model would look something like this (I copy pasted code from a project I used and verified that it worked):

```python
class ResnetGeneratorX(nn.Module):

    def __init__(self, input_nc, style_dim, output_nc, ngf=64):
        super(ResnetGeneratorX, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        downblocks = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            downblocks += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling

        self.resblock_1 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_2 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_3 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_4 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_5 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_6 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_7 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_8 = ResnetBlockX(ngf * mult, style_dim)
        self.resblock_9 = ResnetBlockX(ngf * mult, style_dim)

        upblocks = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            upblocks += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,
                                stride=1, padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        upblocks += [nn.ReflectionPad2d(3)]
        upblocks += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upblocks += [nn.Tanh()]

        self.downblocks = nn.Sequential(*downblocks)
        self.upblocks = nn.Sequential(*upblocks)

    def forward(self, input, style):
        out = self.downblocks(input)
        out = self.resblock_1(out, style)
        out = self.resblock_2(out, style)
        out = self.resblock_3(out, style)
        out = self.resblock_4(out, style)
        out = self.resblock_5(out, style)
        out = self.resblock_6(out, style)
        out = self.resblock_7(out, style)
        out = self.resblock_8(out, style)
        out = self.resblock_9(out, style)
        out = self.upblocks(out)
        return out
```


##### Code

```python
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
```

### Init Weights
|[Github Source](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)|
|-|


##### Usage
You can use the method on any instantiated model.

```python
net = ResNet18()
init_weights(net, init_type='normal', gain=0.01)
```

##### Code

```python
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
```

### Label to one-hot

```python
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
```


### GAN Loss
|[Github Source](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)|
|-|


```python
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (string)-- the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image
            target_fake_label (bool)-- label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) -- tpyically the prediction from a discriminator
            target_is_real (bool) -- if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) -- tpyically the prediction from a discriminator
            target_is_real (bool) -- if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).cuda()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
```

### Calculate Gradient penalty

```python
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD -- discrimiantor network
        real_data -- real images
        fake_data -- generated images from the generator
        device    -- GPU/CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type      -- if we mix real and fake data [real | fake | mixed].
        constant  -- the constant used in formula (||gradient||_2 - constant)^2
        lambda_gp -- weight for this loss
    Returns:
        the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
```

### Perceptual Loss (VGG19)
|[Github Source](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)|
|-|

##### Usage

**Note: This loss has parameters and should therefore be put on GPU if you have a model on GPU.**

We can use the module as a new criterion:

```python
# define the criterion
criterion_VGG = VGGLoss()

# put it on GPU if you can
criterion_VGG = criterion_VGG.cuda()

# calc perceptual loss during train loop
# to compute the perceptual loss of an auto-encoder
# fake_ae is the output of your auto-encoder
# img is the original input image
ae_loss_VGG = criterion_VGG(fake_ae, img)

# do backward or sum up with other losses...
ae_loss_VGG.backward()

```

##### Code

```python
class VGGLoss(nn.Module):
  def __init__(self, gpu_ids=[]):
    super(VGGLoss, self).__init__()
    self.vgg = Vgg19().cuda()
    self.criterion = nn.L1Loss()
    self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

  def forward(self, x, y):
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    loss = 0
    for i in range(len(x_vgg)):
      loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
    return loss


class Vgg19(torch.nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg19, self).__init__()
    vgg_pretrained_features = models.vgg19(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    for x in range(2):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(2, 7):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(7, 12):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])
    for x in range(12, 21):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])
    for x in range(21, 30):
      self.slice5.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, X):
    h_relu1 = self.slice1(X)
    h_relu2 = self.slice2(h_relu1)
    h_relu3 = self.slice3(h_relu2)
    h_relu4 = self.slice4(h_relu3)
    h_relu5 = self.slice5(h_relu4)
    out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
    return out
```

### Self-Attention Layer
|[Github Source](https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py)|[Original Paper](https://arxiv.org/abs/1805.08318)|
|-|-|

##### Usage
You can add it to an existing model as a new layer.

DCGAN Discriminator with Spectral Norm:
```pyton
class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Number of feature in first layer
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.main(x)
```

with additional self-attention after the first conv layer layer:
```python
class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Number of feature in first layer
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SelfAttention(ndf),
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.main(x)
```
##### Code

```python
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
```
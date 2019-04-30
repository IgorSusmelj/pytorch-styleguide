# A PyTorch Tools, best practices & Styleguide
This is not an official style guide for PyTorch. This document summarizes best practices from more than a year of experience with deep learning using the PyTorch framework. Note that the learnings we share come mostly from a research and startup perspective.

This is an open project and other collaborators are highly welcomed to edit and improve the document.

You will find three main parts of this doc. First, a quick recap of best practices in Python, followed by some tips and recommendations using PyTorch. Finally, we share some insights and experiences using other frameworks which helped us generally improve our workflow.


**Update 30.4.2019**
>After so much positive feedback I also added a summary of commonly used building blocks from our projects at [Mirage](https://mirage.id/):
You will find building blocks for (Self-Attention, Perceptual Loss using VGG, Spectral Normalization, Adaptive Instance Normalization, ...)
<br>[Code Snippets for Losses, Layers and other building blocks](building_blocks.md)


## We recommend using Python 3.6+
From our experience we recommend using Python 3.6+ because of the following features which became very handy for clean and simple code:
* [Support for typing since Python 3.6.](https://medium.com/@ageitgey/learn-how-to-use-static-type-checking-in-python-3-6-in-10-minutes-12c86d72677b)
* [Support of f strings since Python 3.6](https://realpython.com/python-f-strings/)


## Python Styleguide recap
We try to follow the Google Styleguide for Python.
Please refer to the well-documented  [style guide on python code provided by Google](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

We provide here a summary of the most commonly used rules:

### Naming Conventions
*From 3.16.4*

| Type | Convention | Example |
|------|------------|---------|
| Packages & Modules | lower_with_under | from **prefetch_generator** import BackgroundGenerator |
| Classes | CapWords | class **DataLoader** |
| Constants | CAPS_WITH_UNDER | **BATCH_SIZE=16** |
| Instances | lower_with_under | **dataset** = Dataset |
| Methods & Functions | lower_with_under() | def **visualize_tensor()** |
| Variables | lower_with_under | **background_color='Blue'** |

## IDEs

### Code Editors
In general, we recommend the use of an IDE such as visual studio code or PyCharm. Whereas VS Code provides syntax highlighting and autocompletion in a relatively lightweight editor PyCharm has lots of advanced features for working with remote clusters.

#### Setting up PyCharm to work with a Remote Machine
1. Login to your remote machine (AWS, Google etc.)
2. Create a new folder and a new virtual environment
3. In Pycharm (professional edition) in the project settings setup a remote interpreter
4. Configure the remote python interpreter (path to venv on AWS, Google etc.)
5. Configure the mapping of the code from your local machine to the remote machine

If set up properly this allows you to do the following:
* Code on your local computer (notebook, desktop) wherever you want (offline, online)
* Sync local code with your remote machine
* Additional packages will be installed automatically on a remote machine
* You don't need any dataset on your local machine
* Run the code and debug on the remote machine as if it would be your local machine running the code


## Jupyter Notebook vs Python Scripts
In general, we recommend to use jupyter notebooks for initial exploration/ playing around with new models and code.
Python scripts should be used as soon as you want to train the model on a bigger dataset where also reproducibility is more important.

**Our recommended workflow:**
1. Start with a jupyter notebook
2. Explore the data and models
3. Build your classes/ methods inside cells of the notebook
4. Move your code to python scripts
5. Train/ deploy on server


| **Jupyter Notebook** | **Python Scripts** |
|----------------------|--------------------|
| + Exploration | + Running longer jobs without interruption |
| + Debugging | + Easy to track changes with git |
| - Can become a huge file| - Debugging mostly means rerunning the whole script|
| - Can be interrupted (don't use for long training) | |
| - Prone to errors and become a mess | |


## Libraries

Commonly used libraries:

| Name | Description | Used for |
|------|-------------|----------|
| [torch](https://pytorch.org/) | Base Framework for working with neural networks | creating tensors, networks and training them using backprop |
| [torchvision](https://pytorch.org/docs/stable/torchvision) | todo | data preprocessing, augmentation, postprocessing |
| [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/) | Python Imaging Library | Loading images and storing them |
| [Numpy](https://www.numpy.org/) | Package for scientific computing with Python | Data preprocessing & postprocessing |
| [prefetch_generator](https://pypi.org/project/prefetch_generator/) | Library for background processing | Loading next batch in background during computation |
| [tqdm](https://github.com/tqdm/tqdm) | Progress bar | Progress during training of each epoch |
| [torchsummary](https://github.com/sksq96/pytorch-summary) | Keras summary for PyTorch | Displays network, it's parameters and sizes at each layer |
| [tensorboardx](https://github.com/lanpa/tensorboardX) | Tensorboard without tensorflow | Logging experiments and showing them in tensorboard |


## File Organization
Don't put all layers and models into the same file. A best practice is to separate the final networks into a separate file (*networks.py*) and keep the layers, losses, and ops in respective files (*layers.py*, *losses.py*, *ops.py*). The finished model (composed of one or multiple networks) should be reference in a file with its name (e.g. *yolov3.py*, *DCGAN.py*)

The main routine, respective the train and test scripts should only import from the file having the model's name.

## Building a Neural Network in PyTorch
We recommend breaking up the network into its smaller reusable pieces. A network is a **nn.Module** consisting of operations or other **nn.Module**s as building blocks. Loss functions are also **nn.Module** and can, therefore, be directly integrated into the network.

A class inheriting from **nn.Module** must have a *forward* method implementing the forward pass of the respective layer or operation. 

A **nn.module** can be used on input data using **self.net(input)**. This simply uses the *__call__()* method of the object to feed the input through the module.

``` python
output = self.net(input)
```

### A Simple Network in PyTorch
Use the following pattern for simple networks with a single input and single output:
``` python
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        block = [nn.Conv2d(...)]
        block += [nn.ReLU()]
        block += [nn.BatchNorm2d(...)]
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)

class SimpleNetwork(nn.Module):
    def __init__(self, num_resnet_blocks=6):
        super(SimpleNetwork, self).__init__()
        # here we add the individual layers
        layers = [ConvBlock(...)]
        for i in range(num_resnet_blocks):
            layers += [ResBlock(...)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
```

Note the following:
* We reuse simple, recurrent building blocks such as *ConvBlock* which consists of the same recurrent pattern of (convolution, activation, normalization) and put them into a separate nn.Module
* We build up a list of desired layers and finally turn them into a model using *nn.Sequential()*. We use the * operator before the list object to unwrap it.
* In the forward pass we just run the input through the model

### A Network with skip connections in PyTorch
``` python
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(...)

    def build_conv_block(self, ...):
        conv_block = []

        conv_block += [nn.Conv2d(...),
                       norm_layer(...),
                       nn.ReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(...)]
            
        conv_block += [nn.Conv2d(...),
                       norm_layer(...)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
```

Here the skip connection of a *ResNet block* has been implemented directly in the forward pass. PyTorch allows for dynamic operations during the forward pass. 

### A Network with multiple outputs in PyTorch
For a network requiring multiple outputs, such as building a perceptual loss using a pretrained VGG network we use the following pattern:
``` python
class Vgg19(nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg19, self).__init__()
    vgg_pretrained_features = models.vgg19(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()

    for x in range(7):
        self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(7, 21):
        self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(21, 30):
        self.slice3.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
        for param in self.parameters():
            param.requires_grad = False

  def forward(self, x):
    h_relu1 = self.slice1(x)
    h_relu2 = self.slice2(h_relu1)        
    h_relu3 = self.slice3(h_relu2)        
    out = [h_relu1, h_relu2, h_relu3]
    return out
```
Note here the following:
* We use a pretrained model provided by *torchvision*.
* We split up the network into three slices. Each slice consists of layers from the pretrained model.
* We *freeze* the network by setting *requires_grad = False*
* We return a list with the three outputs of our slices

## Custom Loss
Even if PyTorch already has a lot of of standard loss function it might be necessary sometimes to create your own loss function. For this, create a separate file `losses.py` and extend the `nn.Module` class to create your custom loss function:

```python
class CustomLoss(nn.Module):
    
    def __init__(self):
        super(CustomLoss,self).__init__()
        
    def forward(self,x,y):
        loss = torch.mean((x - y)**2)
        return loss
```

## Recommended code structure for training your model
Note that we used the following patterns:
* We use *BackgroundGenerator* from *prefetch_generator* to load next batches in background  [see this issue for more information](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5)
* We use tqdm to monitor training progress and show the *compute efficiency*. This helps us find bottlenecks in our data loading pipeline.

``` python
# import statements
import torch
import torch.nn as nn
from torch.utils import data
...

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
...

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ...")
    ...
    opt = parser.parse_args() 
    
    # add code for datasets (we always use train and validation/ test set)
    data_transforms = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(opt.path_to_data, "train"),
        transform=data_transforms)
    train_data_loader = data.DataLoader(train_dataset, ...)
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(opt.path_to_data, "test"),
        transform=data_transforms)
    test_data_loader = data.DataLoader(test_dataset ...)
    ...
    
    # instantiate network (which has been imported from *networks.py*)
    net = MyNetwork(...)
    ...
    
    # create losses (criterion in pytorch)
    criterion_L1 = torch.nn.L1Loss()
    ...
    
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        ...
    
    # create optimizers
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr)
    ...
    
    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint) # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")
        ...
        
    # if we want to run experiment on multiple GPUs we move the models there
    net = torch.nn.DataParallel(net)
    ...
    
    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter(...)
    
    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, opt.epochs):
        # set models to train mode
        net.train()
        ...
        
        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
                    total=len(train_data_loader))
        start_time = time.time()
        
        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            ...
            
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time-time.time()
            
            # forward and backward pass
            optim.zero_grad()
            ...
            loss.backward()
            optim.step()
            ...
            
            # udpate tensorboardX
            writer.add_scalar(..., n_iter)
            ...
            
            # compute computation time and *compute_efficiency*
            process_time = start_time-time.time()-prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time/(process_time+prepare_time), epoch, opt.epochs))
            start_time = time.time()
            
        # maybe do a test pass every x epochs
        if epoch % x == x-1:
            # bring models to evaluation mode
            net.eval()
            ...
            #do some tests
            pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader, ...)),
                    total=len(test_data_loader)) 
            for i, data in pbar:
                ...
                
            # save checkpoint if needed
            ...
```

## Training on Multiple GPUs in PyTorch
There are two distinct patterns in PyTorch to use multiple GPUs for training.
From our experience both patterns are valid. The first one results however in nicer and less code. The second one seems to have a slight performance advantage due to less communication between the GPUs. [I asked a question in the official PyTorch forum about the two approaches here](https://discuss.pytorch.org/t/how-to-best-use-dataparallel-with-multiple-models/39289)

### Split up the batch input of each network
The most common one is to simply split up the batches of all *networks* to the individual GPUs. 
> A model running on 1 GPU with batch size 64 would, therefore, run on 2 GPUs with each a batch size of 32. This can be done automatically by wrapping the model by **nn.DataParallel(model)**.

### Pack all networks in a *super* network and split up input batch
This pattern is less commonly used. A repository implementing this approach is shown here in the [pix2pixHD implementation by Nvidia](https://github.com/NVIDIA/pix2pixHD)


## Do's and Don't's
### Avoid Numpy Code in the forward method of a nn.Module
Numpy runs on the CPU and is slower than torch code. Since torch has been developed with being similar to numpy in mind most numpy functions are supported by PyTorch already.

### Separate the DataLoader from the main Code
The data loading pipeline should be independent of your main training code. PyTorch uses background workers for loading the data more efficiently and without disturbing the main training process.

### Don't log results in every step
Typically we train our models for thousands of steps. Therefore, it is enough to log loss and other results every n'th step to reduce the overhead. Especially, saving intermediary results as images can be costly during training.

### Use Command-line Arguments
It's very handy to use command-line arguments to set parameters during code execution (*batch size*, *learning rate*, etc). An easy way to keep track of the arguments for an experiment is by just printing the dictionary received from *parse_args*:
``` python
...
# saves arguments to config.txt file
opt = parser.parse_args()
with open("config.txt", "w") as f:
    f.write(opt.__str__())
...
```

### Use **.detach()** to free tensors from the graph if possible
PyTorch keeps track of of all operations involving tensors for automatic differentiation. Use **.detach()** to prevent recording of unnecessary operations.

### Use **.item()** for printing scalar tensors
You can print variables directly, however it's recommended to use **variable.detach()** or **variable.item()**. In earlier PyTorch versions < 0.4 you have to use **.data** to access the tensor of a variable.

### Use the call method instead of forward on a **nn.Module**
The two ways are not identical as pointed out in one of the issues [here](https://github.com/IgorSusmelj/pytorch-styleguide/issues/3):
``` python
output = self.net.forward(input)
# they are not equal!
output = self.net(input)
```

## FAQ
1. How to keep my experiments reproducible?
> We recommend setting the following seeds at the beginning of your code:
``` python
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
```
2. How to improve training and inference speed further?
> On Nvidia GPUs you can add the following line at the beginning of our code. This will allow the cuda backend to optimize your graph during its first execution. However, be aware that if you change the network input/output tensor size the graph will be optimized each time a change occurs. This can lead to very slow runtime and out of memory errors. Only set this flag if your input and output have always the same shape. Usually, this results in an improvement of about 20%.
``` python
torch.backends.cudnn.benchmark = True
```
3. What is a good value for compute efficiency using your tqdm + prefetch_generator pattern?
> It depends on the machine used, the preprocessing pipeline and the network size. Running on a SSD on a 1080Ti GPU we see a compute efficiency of almost 1.0 which is an ideal scenario. If shallow (small) networks or a slow harddisk is used the number may drop to around 0.1-0.2 depending on your setup.
4. How can I have a batch size > 1 even though I don't have enough memory?
> In PyTorch we can implement very easily virtual batch sizes. We just prevent the optimizer from making an update of the parameters and sum up the gradients for *batch_size* cycles.
``` python
...
# in the main loop
out = net(input)
loss = criterion(out, label)
# we just call backward to sum up gradients but don't perform step here
loss.backward() 
total_loss += loss.item() / batch_size
if n_iter % batch_size == batch_size-1:
    # here we perform out optimization step using a virtual batch size
    optim.step()
    optim.zero_grad()
    print('Total loss: ', total_loss)
    total_loss = 0.0
...
```
5. How can I adjust the learning rate during training?
> We can access the learning rate directly using the instantiated optimizer as shown here:
``` python
...
for param_group in optim.param_groups:
    old_lr = param_group['lr']
    new_lr = old_lr * 0.1
    param_group['lr'] = new_lr
    print('Updated lr from {} to {}'.format(old_lr, new_lr))
...
```
6. How to use a pretrained model as a loss (non backprop) during training
> If you want to use a pretrained model such as VGG to compute a loss but not train it (e.g. Perceptual loss in style-transfer/ GANs/ Auto-encoder) you can use the following pattern:
``` python
...
# instantiate the model
pretrained_VGG = VGG19(...)

# disable gradients (prevent training)
for p in pretrained_VGG.parameters():  # reset requires_grad
    p.requires_grad = False
...
# you don't have to use the no_grad() namespace but can just run the model
# no gradients will be computed for the VGG model
out_real = pretrained_VGG(input_a)
out_fake = pretrained_VGG(input_b)
loss = any_criterion(out_real, out_fake)
...
```
7. Why do we use *.train()* and *.eval()* in PyTorch?
> Those methods are used to set layers such as **BatchNorm2d** or **Dropout2d** from training to inference mode. Every module which inherits from **nn.Module** has an attribute called *isTraining*. **.eval()** and **.train()** just simply sets this attribute to True/ False. For more information of how this method is implemented please have a look at [the module code in PyTorch](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html)
8. My model uses lots of memory during Inference/ How to run a model properly for inference in PyTorch?
> Make sure that no gradients get computed and stored during your code execution. You can simply use the following pattern to assure that:
``` python
with torch.no_grad():
    # run model here
    out_tensor = net(in_tensor)
```
9. How to fine-tune a pretrained model?
> In PyTorch you can freeze layers. This will prevent them from being updated during an optimization step.
``` python

# you can freeze whole modules using
for p in pretrained_VGG.parameters():  # reset requires_grad
    p.requires_grad = False

```
10. When to use **Variable(...)**?
> Since PyTorch 0.4 **Variable* and **Tensor** have been merged. We don't have to explicitly create a **Variable** object anymore.
11. Is PyTorch on C++ faster then using Python?
> C++ version is about 10% faster
12. Can TorchScript / JIT speed up my code?
> Todo...
13. Is PyTorch code using **cudnn.benchmark=True** faster?
> From our experience you can gain about 20% speed-up. But the first time you run your model it takes quite some time to 
build the optimized graph. In some cases (loops in forward pass, no fixed input shape, if/else in forward, etc.) this flag might
result in *out of memory* or other errors.
14. How to use multiple GPUs for training?
> Todo...
15. How does **.detach()** work in PyTorch?
> If frees a tensor from a computation graph. A nice illustration is shown [here](http://www.bnikolic.co.uk/blog/pytorch-detach.html)


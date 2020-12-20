## Example training cifar10 using PyTorch

This example downloads the cifar10 dataset using torchvision. A simple CNN is 
trained using Adam to about 71% accuracy on the test set.

### Usage

Train a model using default parameters:

```bash
python cifar10_example.py
```

Resume training using a checkpoint:
```bash
python cifar10_example.py --resume --path_to_checkpoint model_checkpoint.ckpt
```
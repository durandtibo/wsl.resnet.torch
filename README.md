# Weakly Supervised Learning of ResNet

This implements weakly supervised learning of residual networks.
This [Torch](http://torch.ch/) implementation is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).s

## Download pretrained models

The `download_pretrained_models` script downloads pretrained models in `data/pretrained_models` directory.

```
th download_pretrained_models.lua
```


## Packages

This implementation uses the following packages:
* torch
* nn
* cunn
* cudnn
* optim
* paths
* csvigo
* matio

You also need to install [spatial-pooling.torch](https://github.com/durandtibo/spatial-pooling.torch) package.

## Training

To train ResNet-101 with WELDON pooling, run `main.lua`
```
th main.lua
```

## License

MIT License

# Weakly Supervised Learning of ResNet

This implements weakly supervised learning of residual networks.
This [Torch](http://torch.ch/) implementation is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

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

To train ResNet-101 with WELDON pooling on VOC 2007 dataset, run `main.lua`
```
th main.lua -optim sgd -LR 1e-3 -netType resnet101-weldon -batchSize 40 -imageSize 224 -data /path_dataset/VOCdevkit/VOC2007/ -dataset voc2007-cls -loss MultiLabel -train multilabel -k 5 -nEpochs 20
```

## License

MIT License

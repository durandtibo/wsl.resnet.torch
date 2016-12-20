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

You also need to install [spatial-pooling.torch](https://github.com/durandtibo/spatial-pooling.torch) package to have spatial pooling modules.

## Training

To train ResNet-101 with WELDON pooling on VOC 2007 dataset, run `main.lua` with
```
th main.lua -optim sgd -LR 1e-2 -netType resnet101-weldon -batchSize 40 -imageSize 224 -data /path_dataset/VOCdevkit/VOC2007/ -dataset voc2007-cls -loss MultiLabel -train multilabel -k 15 -nEpochs 20
```
* `LR`: initial learning rate.
* `imageSize`: size of the image.
* `batchSize`: number of images per batch
* `k`: number of regions for WELDON pooling.
* `nEpochs`: number of training epochs.

To train ResNet-101 with GlobalMaxPooling on VOC 2007 dataset, run `main.lua` with
```
th main.lua -optim sgd -LR 1e-2 -netType resnet101-gap -batchSize 40 -imageSize 224 -data /path_dataset/VOCdevkit/VOC2007/ -dataset voc2007-cls -loss MultiLabel -train multilabel -k 15 -nEpochs 20
```

To train ResNet-101 with GlobalAveragePooling on VOC 2007 dataset, run `main.lua` with
```
th main.lua -optim sgd -LR 1e-2 -netType resnet101-gap -batchSize 40 -imageSize 224 -data /path_dataset/VOCdevkit/VOC2007/ -dataset voc2007-cls -loss MultiLabel -train multilabel -k 15 -nEpochs 20
```

## License

MIT License

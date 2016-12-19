local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 - Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data',         '',            'Path to dataset')
    cmd:option('-dataset',      'voc2007-cls', 'Options: voc2007-cls')
    cmd:option('-manualSeed',   2,             'Manually set RNG seed')
    cmd:option('-backend',      'cudnn',       'Options: cudnn | cunn | nn')
    cmd:option('-cudnn',        'default',     'Options: fastest | default | deterministic')
    cmd:option('-gen',          'gen',         'Path to save generated files')
    cmd:option('-verbose',      'false',       'verbose')
    cmd:option('-save',         'true',        'save model at each epoch')
    ------------- Data options ------------------------
    cmd:option('-nThreads',        1, 'number of data loading threads')
    cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         0,       'Number of total epochs to run')
    cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-imageSize',       224,     'image size')
    cmd:option('-testOnly',        'false', 'Run on test set only')
    cmd:option('-trainOnly',       'false', 'No test')
    ------------- Checkpointing options ---------------
    cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
    cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
    ---------- Optimization options ----------------------
    cmd:option('-optim',          'sgd',  'optimization algorithm')
    cmd:option('-LR',              0.001, 'initial learning rate')
    cmd:option('-LRp',             1,     'learning rate for pretrained layers')
    cmd:option('-momentum',        0.9,   'momentum')
    cmd:option('-weightDecay',     1e-4,  'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',      'resnet',         'Options: resnet | preresnet')
    cmd:option('-netInit',      '',               'Path to inital model')
    cmd:option('-depth',        34,               'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
    cmd:option('-shortcutType', '',               'Options: A | B | C')
    cmd:option('-retrain',      'none',           'Path to model to retrain with')
    cmd:option('-optimState',   'none',           'Path to an optimState to reload from')
    cmd:option('-loss',         'CrossEntropy',   'loss')
    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput',  'false',       'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          'false',       'Use optnet to reduce memory usage')
    cmd:option('-resetClassifier', 'false',       'Reset the fully connected layer for fine-tuning')
    cmd:option('-nClasses',        0,             'Number of classes in the dataset')
    ---------- My options ----------------------------------
    cmd:option('-preprocessing',    'warp',         'image preprocessing')
    cmd:option('-results',          '',             'path to write results')
	cmd:option('-localization',     'false',        'path to write results')
    cmd:option('-train',            'multiclass',   'train with [multiclass | multilabel]')
    cmd:option('-k',                1,              'number of regions in weldon model')
    cmd:option('-alpha',            1.0,            'alpha')
	cmd:option('-features',         '',             'save features')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.verbose = opt.verbose ~= 'false'
    opt.resetClassifier = opt.resetClassifier ~= 'false'
    opt.trainOnly = opt.trainOnly ~= 'false'
    opt.testOnly = opt.testOnly ~= 'false'
    opt.optnet = opt.optnet ~= 'false'
    opt.shareGradInput = opt.shareGradInput ~= 'false'
	opt.localization = opt.localization ~= 'false'

    opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs

	if opt.dataset == 'imagenet' or opt.dataset == 'imagenet2' then
       -- Handle the most common case of missing -data flag
       local trainDir = paths.concat(opt.data, 'train')
       if not paths.dirp(opt.data) then
          cmd:error('error: missing ImageNet data directory')
       elseif not paths.dirp(trainDir) then
          cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
       end
       -- Default shortcutType=B and nEpochs=90
       opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
       opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
    end

    if opt.backend == 'nn' then
        opt.nGPU = 0
    end

    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end

    if opt.resetClassifier then
        if opt.nClasses == 0 then
            cmd:error('-nClasses required when resetClassifier is set')
        end
    end

    if opt.shareGradInput and opt.optnet then
        cmd:error('error: cannot use both -shareGradInput and -optnet')
    end

	opt.pathResults = ''

    return opt
end

return M

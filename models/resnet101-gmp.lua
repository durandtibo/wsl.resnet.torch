
local nn = require 'nn'
require 'spatial-pooling'
local paths = require 'paths'

local function createModel(opt)

    if opt.backend == 'cudnn' then
        require 'cunn'
        require 'cudnn'
    end

    -- load model
    if opt.netInit == '' then
        local filename = 'data/pretrained_models/resnet-101.t7'
        paths.mkdir('data/pretrained_models')
        if not paths.filep(filename) then
            local url = 'https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7'
            os.execute('wget ' .. url .. ' -O ' .. filename)
        end
        opt.netInit = filename
    end
    print('[model] read model: ' .. opt.netInit)
    local pretrainedModel = torch.load(opt.netInit)
    pretrainedModel = pretrainedModel:float()

    -- create model network and remove the last fully-connected layer
    local model = nn.Sequential()
    for i=1,8 do
        model:add(pretrainedModel:get(i))
    end
    if opt.backend == 'cudnn' then
        cudnn.convert(model, nn)
    end
    model = model:float()
    collectgarbage()
    local numPretrainedParam = model:getParameters():size(1)

    -- add new fully-connected layer for target dataset
    local transfert = nn.SpatialConvolution(2048, opt.nClasses, 1, 1, 1, 1)
    transfert.bias:zero()
    model:add(transfert)
    model = model:float()

    -- find the output size to define the spatial aggregation layer
    local input = torch.rand(1, 3, opt.imageSize, opt.imageSize):float()
    print('[model] input ' .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3) .. ' x ' .. input:size(4))
    local output = model:forward(input)
    print('[model] output after convolutions: ' .. output:size(1) .. ' x ' .. output:size(2) .. ' x ' .. output:size(3) .. ' x ' .. output:size(4))

    -- add spatial aggregation layer
    model:add(nn.GlobalMaxPooling())
    model:add(nn.Reshape(opt.nClasses))

    model = model:float()
    local output = model:forward(input)
    print('[model] output ' .. output:size(1) .. ' x ' .. output:size(2))
    local numParam = model:getParameters():size(1)

    opt.lastconv = 9

    if opt.LRp ~= 1 and opt.LRp >= 0 then
        print('[model] initialize learningRates', opt.LRp)
        local lrs = torch.Tensor(numParam):zero()
        for i = 1, numParam do
            if i > numPretrainedParam then
                lrs[i] = 1.0
            else
                lrs[i] = opt.LRp
            end
        end
        opt.learningRates = lrs
        if opt.backend == 'cudnn' then
            opt.learningRates = opt.learningRates:cuda()
        end
    end

    return model
end

return createModel

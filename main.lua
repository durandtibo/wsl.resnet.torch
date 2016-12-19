--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)

-- if GPU, load cutorch
if opt.backend == 'cudnn' or opt.backend == 'cunn' then
    require 'cutorch'
    cutorch.manualSeedAll(opt.manualSeed)
end

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
local trainLoader, testLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

local Trainer
if opt.train == 'multilabel' then
    Trainer = require 'trainMultiLabel'
else
    Trainer = require 'trainMultiClass'
end
-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
    local top1Err, top5Err = trainer:test(0, testLoader)
    print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
    return
end

paths.mkdir('log')
local testLogger = optim.Logger('log/' .. opt.dataset .. '_' .. os.date("%c") .. '.log')
testLogger:setNames({'epoch', 'testTop1', 'testTop5', 'trainTop1', 'trainTop5', 'trainLoss'})

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

    local testTop1, testTop5 = 0
    if not opt.trainOnly then
        -- Run model on validation set
        testTop1, testTop5 = trainer:test(epoch, testLoader)
    end

    local bestModel = false
    if testTop1 <= bestTop1 then
        bestModel = true
        bestTop1 = testTop1
        bestTop5 = testTop5
        print(' * Best model ', testTop1, testTop5)

        if opt.results ~= '' then
            trainer:writeScores(epoch, trainLoader, 'train')
            trainer:writeScores(epoch, testLoader, 'test')
        end
    end

    print(' * Best model ', bestTop1)

    if testLogger then
        testLogger:add{epoch, testTop1, testTop5, trainTop1, trainTop5, trainLoss}
    end

    if opt.save ~= '' then
        checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
    end

end

print(string.format(' * Finished best model: %6.3f', bestTop1))

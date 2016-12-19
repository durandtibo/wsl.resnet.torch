--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local evaluation = require 'evaluation'
local csvigo = require 'csvigo'
local paths = require 'paths'
local matio = require 'matio'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
	self.model = model
	self.criterion = criterion
	self.optimState = optimState or {
		learningRate = opt.LR,
		learningRateDecay = 0.0,
		momentum = opt.momentum,
		nesterov = true,
		dampening = 0.0,
		weightDecay = opt.weightDecay,
	}
	self.opt = opt
	self.params, self.gradParams = model:getParameters()
	self.featureNet = opt.featureNet
end

function Trainer:train(epoch, dataloader)
	-- Trains the model for a single epoch
	self.optimState.learningRate = self:learningRate(epoch)

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()

	local function feval()
		return self.criterion.output, self.gradParams
	end

	local nImages = dataloader:numberOfImages()
	local lossSum = 0.0
	local N = 0

	if self.featureNet ~= nil then
		self.featureNet:evaluate()
	end

	print('=> Training epoch # ' .. epoch)
	-- set the batch norm to training mode
	self.model:training()
	if self.featureNet ~= nil then
		self.featureNet:evaluate()
	end
	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		-- Copy input and target to the GPU
		self:copyInputs(sample)

		local target = self.target:clone()
		target = torch.cmin(target:add(1), 1)

		local input = self.input

		if self.featureNet ~= nil then
			input = self.featureNet:forward(input)
		end

		local output = self.model:forward(input):float()
		local batchSize = output:size(1)
		local loss = self.criterion:forward(self.model.output, target)

		self.model:zeroGradParameters()
		self.criterion:backward(self.model.output, target)
		self.model:backward(input, self.criterion.gradInput)

		optim.sgd(feval, self.params, self.optimState)

		lossSum = lossSum + loss*batchSize
		N = N + batchSize

		print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  loss %7.3f (%7.3f)'):format(
		epoch, n, size, timer:time().real, dataTime, loss, lossSum / N))

		-- check that the storage didn't get changed do to an unfortunate getParameters call
		assert(self.params:storage() == self.model:parameters()[1]:storage())

		timer:reset()
		dataTimer:reset()
	end

	return 0, 0, lossSum / N
end

function Trainer:test(epoch, dataloader)
	-- Computes the top-1 and top-5 err on the validation set

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()

	local nImages = dataloader:numberOfImages()
	local lossSum = 0.0
	local scores = torch.zeros(nImages, self.opt.nClasses)
	local labels = torch.zeros(nImages, self.opt.nClasses)
	local N = 0
	local name = {}

	self.model:evaluate()
	if self.featureNet ~= nil then
		self.featureNet:evaluate()
	end

	for n, sample in dataloader:run() do
		local dataTime = dataTimer:time().real

		-- Copy input and target to the GPU
		self:copyInputs(sample)

		local target = self.target:clone()
		target = torch.cmin(target:add(1), 1)

		local input = self.input

		if self.featureNet ~= nil then
			input = self.featureNet:forward(input)
		end

		local output = self.model:forward(input):float()
		local batchSize = output:size(1)
		local loss = self.criterion:forward(self.model.output, target)

		scores[{{N+1,N+batchSize},{}}] = output
		labels[{{N+1,N+batchSize},{}}] = self.target:float()
		for i = 1,batchSize do
			name[N+i] = sample.path[i]
		end

		lossSum = lossSum + loss*batchSize
		N = N + batchSize

		print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  loss %7.3f (%7.3f)'):format(
		epoch, n, size, timer:time().real, dataTime, loss, lossSum / N))

		timer:reset()
		dataTimer:reset()
	end
	self.model:training()

	map =  Evaluation:computeMeanAveragePrecision(scores, labels)

	print((' * Finished epoch # %d     MAP: %7.3f    error: %7.3f    loss: %7.3f\n'):format(epoch, map, 1-map, lossSum / N))

	return 1-map, lossSum / N
end

function Trainer:writeScores(epoch, dataloader, split)
	-- Computes the top-1 and top-5 err on the validation set

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	local size = dataloader:size()
	local nImages = dataloader:numberOfImages()

	local lossSum = 0.0
	local scores = torch.zeros(nImages, self.opt.nClasses)
	local labels = torch.zeros(nImages, self.opt.nClasses)
	local N = 0

	local pathResults = self.opt.results .. '/classification/' .. self.opt.dataset .. '/' .. self.opt.netType .. '_image_' .. self.opt.imageSize .. self.opt.pathResults .. '/LR=' .. self.opt.LR
	local file = pathResults .. '/scores_' .. split .. '.csv'
	local path = paths.dirname(file)
	print('write scores in ' .. path)

	self.model:evaluate()
	if self.featureNet ~= nil then
		self.featureNet:evaluate()
	end

	local name = {}

	for n, sample in dataloader:run() do
		xlua.progress(n, size)

		-- Copy input and target to the GPU
		self:copyInputs(sample)

		local input = self.input

		local target = self.target:clone()
		target = torch.cmin(target:add(1), 1)

		if self.featureNet ~= nil then
			input = self.featureNet:forward(input)
		end

		local output = self.model:forward(input):float()
		local batchSize = output:size(1)
		local loss = self.criterion:forward(self.model.output, target)

		scores[{{N+1,N+batchSize},{}}] = output
		labels[{{N+1,N+batchSize},{}}] = self.target:float()
		for i = 1,batchSize do
			name[N+i] = sample.path[i]
		end

		if self.opt.localization then
			output = self.model:get(self.opt.lastconv).output

			for i = 1,batchSize do
				local filename = pathResults .. '/localization/' .. sample.path[i] .. '.mat'
				local path = paths.dirname(filename)
				paths.mkdir(path)
				matio.save(filename, output[{i,{},{},{}}]:float())
			end
		end

		N = N + batchSize

	end

	local data = {}
	-- header
	data[1] = {}
	data[1][1] = 'name'
	for j=1,scores:size(2) do
		data[1][j+1] = 'class' .. j
	end
	for i=1,table.getn(name) do
		data[i+1] = {}
		data[i+1][1] = name[i]
		for j=1,scores:size(2) do
			data[i+1][j+1] = scores[i][j]
		end
	end
	paths.mkdir(path)
	csvigo.save({path = file, mode = "raw", data = data})
end

function Trainer:copyInputs(sample)
	-- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
	-- if using DataParallelTable. The target is always copied to a CUDA tensor
	self.input = self.input or (self.opt.nGPU == 0 and torch.FloatTensor() or self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
	self.target = self.target or (self.opt.nGPU == 0 and torch.FloatTensor() or torch.CudaTensor())
	self.input:resize(sample.input:size()):copy(sample.input)
	self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
	-- Training schedule
	local decay = 0
	return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer

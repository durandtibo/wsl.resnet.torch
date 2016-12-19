require 'torch'

local Evaluation = torch.class('Evaluation')

-- remove image with label 0
function Evaluation:removeDifficults(scores, gtlabels)

	local numExamples = (#scores)[1]
	local n = 0;
	for i=1,numExamples do
		if gtlabels[i] ~= 0 then
			n = n + 1
		end
	end

	local newScores = torch.zeros(n)
	local newGtlabels = torch.zeros(n)

	n=0
	for i=1,numExamples do
		if gtlabels[i] ~= 0 then
			n = n + 1
			newScores[n] = scores[i]
			newGtlabels[n] = gtlabels[i]
		end
	end

	return newScores, newGtlabels
end

-- -1 -> neg, 1 -> pos, 0 -> difficult
function Evaluation:computeAveragePrecision(scores, gtlabels)

	scores, gtlabels = Evaluation:removeDifficults(scores, gtlabels)

	--print('scores', #scores, 'gtlabels', #gtlabels)

	local y,si = torch.sort(scores, true)
	--print('y',y,'si',si)

	local numExamples = (#scores)[1]
	--print('numExamples', numExamples)
	local tp = torch.zeros(numExamples)
	local fp = torch.zeros(numExamples)

	for i=1,numExamples do
		if gtlabels[si[i]] > 0 then tp[i] = 1 end
		if gtlabels[si[i]] < 0 then fp[i] = 1 end
	end

	fp = fp:cumsum()
	tp = tp:cumsum()

	local numPos = 0
	for i=1,numExamples do
		if gtlabels[i] > 0 then numPos = numPos + 1 end
	end

	local rec = torch.zeros(numExamples)
	local prec = torch.zeros(numExamples)
	for i=1,numExamples do
		rec[i] = tp[i] / numPos
		prec[i] = tp[i] / (tp[i] + fp[i])
	end


	local mrec = torch.zeros(numExamples + 2)
	local mprec = torch.zeros(numExamples + 2)

	mrec[{{2,numExamples+1}}] = rec
	mrec[numExamples+2] = 1.0
	mprec[{{2,numExamples+1}}] = prec

	for i=numExamples-1,1,-1 do
		mprec[i] = math.max(mprec[i], mprec[i+1])
	end

	local ap = 0.0
	for i=1,numExamples-1 do
		if mrec[i+1] ~= mrec[i] then
			ap = ap + (mrec[i+1] - mrec[i]) * mprec[i+1]
		end
	end

	return ap
end

function Evaluation:computeMeanAveragePrecision(scores, gtlabels)

	local numExamples = (#gtlabels)[1]
	local numClasses = (#gtlabels)[2]
	--print('numClasses',numClasses)
	local apPerClass = torch.zeros(numClasses)
	local map = 0.0
	local n = 0
	for i=1,numClasses do
		local ap = Evaluation:computeAveragePrecision(scores[{{}, {i}}]:reshape(numExamples), gtlabels[{{}, {i}}]:reshape(numExamples))
		apPerClass[i] = ap
		if isnan(ap) == false then
			map = map + ap
			n = n + 1
		end
	end
	print('apPerClass',apPerClass)
	return map / n
end

function isnan(x) return x ~= x end

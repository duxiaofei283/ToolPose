require 'cunn'
require 'cudnn'
require 'cutorch'
local Runner = require 'runner_invivo'

torch.setdefaulttensortype('torch.FloatTensor')

local save_parameters = {'weight', 'bias', 'running_mean', 'running_var', 'running_std' }

local function copyModel(src, dst)
	assert(torch.type(src) == torch.type(dst), 'torch.type(src) ~= torch.type(dst)')
	for i,k in ipairs(save_parameters) do
		local v = src[k]
		if v ~= nil then
			dst[k]:copy(v)
		end
	end
	if src.modules ~= nil then
		assert(#dst.modules == #src.modules, '#dst.modules ~= #src.modules')
		local nModule = #src.modules
		if nModule > 0 then
			for i=1,nModule do
				copyModel(src.modules[i], dst.modules[i])
			end
		end
	end
end

local oldDataDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label'
if not paths.dirp(oldDataDir) then error("Can't find directory : " .. oldDataDir) end

local newDataDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/tool_label'
if not paths.dirp(newDataDir) then error("Can't find directory : " .. newDataDir) end

local saveDir = '/home/xiaofei/workspace/toolPose/models'
if not paths.dirp(saveDir) then
	os.execute('mkdir -p ' .. saveDir)
end

local function getSaveID(modelConf, dataType)
    local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
    if dataType ~= nil then
    	s = s .. '_' .. dataType
	end
    return s
end

local function getDetID(modelConf, dataType)
	local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
	if modelConf.jointRadius ~= nil then
		s = s .. '_r' .. modelConf.jointRadius
	end
	if dataType ~= nil then
    	s = s .. '_' .. dataType
	end
	print(s)
    return s
end

local function getRegressID(modelConf)
    local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
--	if modelConf.jointRadius ~= nil then
--		s = s .. '_r' .. modelConf.jointRadius
--	end
	s = s .. '_whole'
    return s
end

local function getInitID(modelConf)
	local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
    return s
end

local opt = {
	oldDataDir = oldDataDir,
    newDataDir = newDataDir,
	saveDir = saveDir,
	dataType = 'invivo2', -- invivo, icl
	retrain = 'last', -- nil, 'last' or 'best'
	learningRate = 1e-4,  -- old 1e-5
	momentum = 0.98,
	weightDecay = 0.0005, -- old 0.0005
	decayRatio = 0.95,
	updateIternal = 10,
--    detModelConf = {type='toolDualPoseSep', v=1, jointRadius=20, modelOutputScale=4},
--	modelConf = {type='toolPoseRegress', v=1, jointRadius = 20, modelOutputScale=4},

--	detModelConf = {type='toolPartDetFull', v='256*320_ftblr', jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256},
--	modelConf = {type='toolPoseRegressFull', v=2, jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10},
--	modelConf = {type='toolPoseRegressFull', v='256*320_ftblr', jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=1, hflip=1},

--	detModelConf = {type='toolPartDetFull', v='256*320_ftblr_head', jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256},
--	modelConf = {type='toolPoseRegressFull', v='256*320_ftblr_head', jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=1, hflip=1},

	-- larger radius for detection model
	detModelConf = {type='toolPartDetFull', v='256*320_ftblr_head', jointRadius=15, modelOutputScale=1, inputWidth=320, inputHeight=256},
	modelConf = {type='toolPoseRegressFull', v='256*320_ftblr_head_noConcat', jointRadius=20, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=1, hflip=1},

	gpus = {1},
	nThreads = 6,
--	batchSize = 1,
	trainBatchSize = 2,
	valBatchSize = 2,
	rotMaxDegree = 0,
    toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint', 'EndPoint' }, -- joint number = 5
	toolCompoNames = {{'LeftClasperPoint', 'HeadPoint'},
					  {'RightClasperPoint', 'HeadPoint'},
					  {'HeadPoint', 'ShaftPoint'},
                      {'ShaftPoint', 'EndPoint'}
					 },
	nEpoches = 300
}
opt.inputWidth = opt.modelConf.inputWidth or 320 -- 480  -- 720
opt.inputHeight = opt.modelConf.inputHeight or 256 -- 384 -- 576
opt.modelOutputScale = opt.modelConf.modelOutputScale or 4
opt.detJointRadius = opt.detModelConf.jointRadius or 10
opt.jointRadius = opt.modelConf.jointRadius or 20
opt.normalScale = opt.modelConf.normalScale or 1
opt.vflip = opt.modelConf.vflip or 0
opt.hflip = opt.modelConf.hflip or 0

local detID = getDetID(opt.detModelConf, opt.dataType)
local detModelPath = paths.concat(opt.saveDir, 'model.' .. detID .. '.best.t7')

local initID = getRegressID(opt.modelConf)
local saveID = getSaveID(opt.modelConf, opt.dataType)
local initModelPath = paths.concat(opt.saveDir, 'model.' .. initID .. '.best.t7')
local lastModelPath = paths.concat(opt.saveDir, 'model.' .. saveID .. '.last.t7')
local lastOptimStatePath = paths.concat(opt.saveDir, 'optim.' .. saveID .. '.last.t7')
local bestModelPath = paths.concat(opt.saveDir, 'model.' .. saveID .. '.best.t7')
local bestOptimStatePath = paths.concat(opt.saveDir, 'optim.' .. saveID .. '.best.t7')
local loggerPath = paths.concat(opt.saveDir, 'log.' .. saveID .. '.t7')
local logPath = paths.concat(opt.saveDir, 'log.' .. saveID .. '.txt')

local function getDetModelPath()
    local modelPath
--	print(detModelPath)
    if paths.filep(detModelPath) then
        modelPath = detModelPath
	end
    print('current using detection model: ' .. modelPath)
    return modelPath
end

local function getModelPath()
    local modelPath
    if opt.retrain == 'last' and paths.filep(lastModelPath) then
        modelPath = lastModelPath
    elseif opt.retrain == 'best' and paths.filep(bestModelPath) then
        modelPath = bestModelPath
    else
        modelPath = initModelPath
	end
	print('current using model: ' .. modelPath)
    return modelPath
end

local function getModel()
    local model = torch.load(getModelPath())
    return model
end

local function getOptimState()
	local optimState
	if opt.retrain == 'last' and paths.filep(lastOptimStatePath) then
		optimState = torch.load(lastOptimStatePath)
--		optimState.learningRate = 1e-5
	elseif opt.retrain == 'best' and paths.filep(bestOptimStatePath) then
		optimState = torch.load(bestOptimStatePath)
	else
		optimState = {
			learningRate = opt.learningRate,
			weightDecay = opt.weightDecay,
			momentum = opt.momentum,
			dampening = 0.0,
			nesterov = true,
			epoch = 0
		}
	end
	return optimState
end

-- when saving, clear the potential tensors in the optim state
local function saveOptimState(save_path, optim_state)
	local optimState = {}
	for key, value in pairs(optim_state) do
		if not torch.isTensor(value) then
			optimState[key] = value
		end
	end
	torch.save(save_path, optimState)
end

local detModel_path = getDetModelPath()
local model_path = getModelPath()
local model
local runningState = {valAcc=1e+8, model = getModel(), optimState = getOptimState() }

-- The runner handles the training loop and evaluate on the val set
local runner = Runner(detModel_path, model_path, opt, runningState.optimState)
model = runner:getModel()
print('optim State: ')
print(runningState.optimState)
local best_epoch = runningState.optimState.epoch
local logFile = io.open(logPath, 'w')
local logger = torch.FloatTensor(opt.nEpoches, 5)

-- Run model on validation set
local valAcc, valLoss, oldValPrec, newValPrec, oldValAcc, newValAcc = runner:val(0)
print(string.format("Val : robustness accuracy = %.3f, loss = %.5f", valAcc, valLoss))
print(string.format('Old: robustness accurary = %.3f', oldValAcc))
print(string.format('New: robustness accurary = %.3f', newValAcc))
print(string.format("Old : precision distance = %.3f", oldValPrec))
print(string.format("New : precision distance = %.3f", newValPrec))


for epoch = 1, opt.nEpoches do
    print('\nepoch # ' .. epoch)

    -- train for a single epoch
    local trainAcc, trainLoss, oldTrainPrec, newTrainPrec, oldTrainAcc, newTrainAcc = runner:train(epoch)
    print(string.format("Train : robustness accuracy = %.3f, loss = %.5f", trainAcc, trainLoss))
	print(string.format('Old: robustness accurary = %.3f', oldTrainAcc))
	print(string.format('New: robustness accurary = %.3f', newTrainAcc))
	print(string.format("Old : precision distance = %.3f", oldTrainPrec))
    print(string.format("New : precision distance = %.3f", newTrainPrec))


	-- Run model on validation set
    local valAcc, valLoss, oldValPrec, newValPrec = runner:val(epoch)
    print(string.format("Val : robustness accuracy = %.3f, loss = %.5f", valAcc, valLoss))
	print(string.format("Val : OLD precision distance = %.3f", oldValPrec))
    print(string.format("Val : NEW precision distance = %.3f", newValPrec))

    local testAcc, testLoss = runner:test(epoch)
	print(string.format("Test : test random Sample."))

--	copyModel(model, runningState.model)
--	torch.save(lastModelPath, runningState.model)
	torch.save(lastModelPath, model:clearState())
	saveOptimState(lastOptimStatePath, runningState.optimState)

	logger[epoch][1] = runningState.optimState.epoch
	logger[epoch][2] = trainAcc
	logger[epoch][3] = valAcc
	logger[epoch][4] = trainLoss
	logger[epoch][5] = valLoss
	logFile:write(string.format('%d %.3f %.3f %.5f %.5f\n',
	logger[epoch][1], logger[epoch][2], logger[epoch][3], logger[epoch][4], logger[epoch][5]))
	logFile:flush()
	torch.save(loggerPath, logger)


	print('optim State for this epoch: ')
	print(runningState.optimState)

	print(string.format("Train : robustness accuracy = %.3f, loss = %.5f", trainAcc, trainLoss))
	print(string.format("Old : precision distance = %.3f", oldTrainPrec))
	print(string.format('New: precision distance = %.3f', newTrainPrec))
	print(string.format("Val : robustness accuracy = %.3f, loss = %.5f", valAcc, valLoss))
	print(string.format("Old : precision distance = %.3f", oldValPrec))
	print(string.format("New : precision distance = %.3f", newValPrec))

	if newValPrec < runningState.valAcc then
		print('Saving the best! ')
		best_epoch = runningState.optimState.epoch
		runningState.valAcc = newValPrec
--		torch.save(bestModelPath, runningState.model)
		torch.save(bestModelPath, model:clearState())
		saveOptimState(bestOptimStatePath, runningState.optimState)
    end
end
logFile:write(string.format('bestModel.epoch = %d, bestModel.valAcc = %.3f', best_epoch, runningState.valAcc))
logFile:flush()
logFile:close()

-- copy the log file
local logFinalPath = paths.concat(opt.saveDir, 'log.' .. saveID .. '_ep' .. runningState.optimState.epoch .. '.txt')
local inlogFile = io.open(logPath, 'r')
local instr = inlogFile:read('*a')
inlogFile:close()
local outlogFile = io.open(logFinalPath, 'w')
outlogFile:write(instr)
outlogFile:close()

logger = nil
runningState.model = nil
runningState.optimState = nil
model = nil


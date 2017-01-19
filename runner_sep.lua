local optim = require('optim')
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'xlua'
require 'image'

local DataLoader = require 'dataloader_sep'

torch.setdefaulttensortype('torch.FloatTensor')

-- frames       [b,c=3,H,W]
-- gt_jtmaps    [b,c=1,h,w]
-- outputs      [b*h*w,7]
local function visualResult(frames, gt_jtmaps, outputs, joint_names, saveDir)
    local batch_size = gt_jtmaps:size(1)
    local outputs_map = torch.round(outputs)
    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), frames[rand_idx]:byte())
    for i=1, #joint_names do
        image.save(paths.concat(saveDir, string.format('frame_%s_gt.png', joint_names[i])), gt_jtmaps[rand_idx][i]:byte())
        image.save(paths.concat(saveDir, string.format('frame_%s_result.png', joint_names[i])), outputs_map[rand_idx][i]:byte())
    end
end

local M = {}
local Runner = torch.class('Runner', M)
function Runner:__init(net_path, opt, optimState)
    -- load network
    print('Loading network ...')
    self.model = torch.load(net_path)
    print(self.model)
    self.model:cuda()

    -- opt
    self.opt = opt
    self.optimState = optimState
    self.trainBatchSize = opt.trainBatchSize or opt.batchSize or 1
    self.valBatchSize = opt.valBatchSize or opt.batchSize or 1
    self.batchSize = opt.batchSize or 1
    self.toolJointNames = opt.toolJointNames

    self.nGPU = #opt.gpus
    local nGPU = #opt.gpus
    if nGPU > 1 then
        print('converting module to nn.DataParallelTable')
        assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
		local dpt = nn.DataParallelTable(1, true, true):add(self.extractModel, opt.gpus):threads(function()
			require('cudnn')
		end)
		dpt.gradInput = nil
		self.model = dpt
    else
        cudnn.fastest = true
        cudnn.benchmark = true
    end

    -- data related
    self.dataLoader = DataLoader(opt)
    self.inputWidth = opt.inputWidth
    self.inputHeight = opt.inputHeight

    self.framesGPU = nil
    self.jointmapsGPU = nil
    if nGPU > 1 then
        self.framesGPU = cutorch.createCudaHostTensor(self.batchSize, 3, self.inputHeight, self.inputWidth)
        self.jointmapsGPU = cutorch.createCudaHostTensor(self.batchSize, 1, self.inputHeight, self.inputWidth)
    else
--        self.framesGPU = torch.CudaTensor(self.batchSize, 3, self.inputHeight, self.inputWidth)
--        self.jointmapsGPU = torch.CudaTensor(self.batchSize, 1, self.inputHeight, self.inputWidth)
        self.framesGPU = torch.CudaTensor()
        self.jointmapsGPU = torch.CudaTensor()
    end

    self.params, self.gradParams = self.model:getParameters()
    print('model #params = ' .. tostring(#self.params))

    self.criterion = nn.BCECriterion()
    self.criterion:cuda()

end

function Runner:getModel()
    return self.model
end

function Runner:train(epoch)
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local dataTime = 0
    local size = self.dataLoader:trainSize()
    local loss = 0.0
    local acc = 0.0
    local N = 0

    self.model:training()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    for n, framesCPU, jointmapsCPU in self.dataLoader:load(1) do
        -- load data
        dataTime = dataTime + dataTimer:time().real
        -- transfer over to GPU
        self.framesGPU:resize(framesCPU:size()):copy(framesCPU)
        self.jointmapsGPU:resize(jointmapsCPU:size()):copy(jointmapsCPU)

        -- reset gradparameters
        self.gradParams:zero()
        -- forward
--        print(self.framesGPU:size())
--        print(self.jointmapsGPU:size())
        local outputsGPU = self.model:forward(self.framesGPU)
        local loss_batch = self.criterion:forward(outputsGPU, self.jointmapsGPU)

        loss = loss + loss_batch
        -- backward
        local grad_output = self.criterion:backward(outputsGPU, self.jointmapsGPU)
        self.model:backward(self.framesGPU, grad_output)
        -- update parameters
        optim.sgd(feval, self.params, self.optimState)
        -- accumulate accuracy
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.jointmapsGPU)):sum() / self.jointmapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == 3 then
            visualResult(framesCPU, jointmapsCPU, outputsGPU, self.toolJointNames, '/home/xiaofei/workspace/toolPose/sep_results/train')
        end

        -- check that the storage didn't get changed due to an unfortunate getParameters call
        assert(self.params:storage() == self.model:parameters()[1]:storage())
        xlua.progress(n, size)
        collectgarbage()
        collectgarbage()
        dataTimer:reset()
    end
    -- update optimState
    self.optimState.epoch = self.optimState.epoch + 1
    if epoch % self.opt.updateIternal == 0 then
        self.optimState.learningRate = self.optimState.learningRate * self.opt.decayRatio
        self.optimState.weightDecay = self.optimState.weightDecay * self.opt.decayRatio
    end
    -- calculate loss, acc
    loss = loss / N
    acc = acc * 100 / N
    print("\nTrain : time to learn = " .. timer:time().real .. ' sec')
	print("Train : time to load data = " .. dataTime .. ' sec')
    return acc, loss


end

function Runner:val(epoch)
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local dataTime = 0
    local size = self.dataLoader:valSize()
    local loss = 0.0
    local acc = 0.0
    local N = 0

    self.model:evaluate()

    for n, framesCPU, jointmapsCPU in self.dataLoader:load(2) do
        -- load data
        dataTime = dataTime + dataTimer:time().real
        -- transfer over to GPU
        self.framesGPU:resize(framesCPU:size()):copy(framesCPU)
        self.jointmapsGPU:resize(jointmapsCPU:size()):copy(jointmapsCPU)

        -- forward
--        print(self.framesGPU:size())
--        print(self.jointmapsGPU:size())
        local outputsGPU = self.model:forward(self.framesGPU)
        local loss_batch = self.criterion:forward(outputsGPU, self.jointmapsGPU)
        loss = loss + loss_batch
        -- accumulate accuracy
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.jointmapsGPU)):sum() / self.jointmapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == 3 then
            visualResult(framesCPU, jointmapsCPU, outputsGPU, self.toolJointNames, '/home/xiaofei/workspace/toolPose/sep_results/val')
        end

        xlua.progress(n, size)
        collectgarbage()
        collectgarbage()
        dataTimer:reset()
    end

    -- calculate loss, acc
    loss = loss / N
    acc = acc * 100 / N
    print("\nVal : time to predict = " .. timer:time().real .. ' sec')
	print("Val : time to load data = " .. dataTime .. ' sec')
    return acc, loss

end

return M.Runner
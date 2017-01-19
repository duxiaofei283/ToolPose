local optim = require('optim')
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'xlua'
require 'image'

local DataLoader = require 'dataloader_aug'

torch.setdefaulttensortype('torch.FloatTensor')

-- frames       [b,c=3,H,W]
-- gt_jtmaps    [b,c=1,h,w]
-- outputs      [b*h*w,7]
local function visualResult(frames, gt_jtmaps, outputs, saveDir)
    local batch_size = gt_jtmaps:size(1)
    local outputmap_height = gt_jtmaps:size(3)
    local outputmap_width = gt_jtmaps:size(4)

    local __, batch_max_indice = torch.max(outputs:view(-1,7), 2)
    batch_max_indice = batch_max_indice:byte()
    batch_max_indice = batch_max_indice:reshape(batch_size, outputmap_height, outputmap_width, 1)
    local output_jtmaps = batch_max_indice:permute(1,4,2,3)    --[b,c=1,h,w]

    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), frames[rand_idx]:byte())
    image.save(paths.concat(saveDir, 'frame_gt.png'), gt_jtmaps[rand_idx]:byte())
    image.save(paths.concat(saveDir, 'frame_result.png'), output_jtmaps[rand_idx]:byte())
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

    self.criterion = nn.CrossEntropyCriterion()
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
        local loss_batch = self.criterion:forward(outputsGPU,
                                                  self.jointmapsGPU:view(self.jointmapsGPU:nElement()))

        loss = loss + loss_batch
        -- backward
        local grad_output = self.criterion:backward(outputsGPU, self.jointmapsGPU:view(self.jointmapsGPU:nElement()))
        self.model:backward(self.framesGPU, grad_output)
        -- update parameters
        optim.sgd(feval, self.params, self.optimState)
        -- accumulate accuracy
        local __, batch_max_indice = torch.max(outputsGPU, 2)
        local batch_acc = torch.eq(batch_max_indice:cuda(), self.jointmapsGPU:view(self.jointmapsGPU:nElement())):sum() / self.jointmapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == 3 then
            visualResult(framesCPU, jointmapsCPU, outputsGPU, '/home/xiaofei/workspace/toolPose/results/train')
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
        local loss_batch = self.criterion:forward(outputsGPU:view(-1, 7),
                                                  self.jointmapsGPU:view(-1))

        loss = loss + loss_batch
        -- accumulate accuracy
        local __, batch_max_indice = torch.max(outputsGPU:view(-1,7), 2)
        local batch_acc = torch.eq(batch_max_indice:cuda(), self.jointmapsGPU:view(-1)):sum() / self.jointmapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == 3 then
            visualResult(framesCPU, jointmapsCPU, outputsGPU, '/home/xiaofei/workspace/toolPose/results/val')
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
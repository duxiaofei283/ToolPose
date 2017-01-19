local optim = require('optim')
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'xlua'
require 'image'
local matio = require 'matio'
local DataLoader = require 'dataloader_dual'

torch.setdefaulttensortype('torch.FloatTensor')

local function saveMatResult(frames, gt_maps, outputs_map, joint_names, compo_names, saveDir)
    outputs_map:clamp(0,1)
    local batch_size = frames:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    local saved_mat = {}
    saved_mat['frame'] = frames[rand_idx]:float()
    local joint_num = #joint_names
    for i=1, joint_num do
        if gt_maps ~= nil then saved_mat[string.format('conf_%s_gt', joint_names[i])] = gt_maps[rand_idx][i]:float() end
        saved_mat[string.format('conf_%s_result', joint_names[i])] = outputs_map[rand_idx][i]:float()
    end

    local compo_num = #compo_names
    for i=1, compo_num do
        if gt_maps ~= nil then saved_mat[string.format('compo_%s_%s_gt', compo_names[i][1], compo_names[i][2])] = gt_maps[rand_idx][joint_num+i]:float() end
        saved_mat[string.format('compo_%s_%s_result', compo_names[i][1], compo_names[i][2])] = outputs_map[rand_idx][joint_num+i]:float()
    end
    matio.save(paths.concat(saveDir, 'output.mat'), saved_mat)
end
-- frames       [b,c=3,H,W]
-- gt_jtmaps    [b,c=1,h,w]
-- outputs      [b*h*w,9]
local function visualResult(frames, gt_maps, outputs_map, joint_names, compo_names, saveDir)
    local batch_size = frames:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), (255*frames[rand_idx]):byte())
    local joint_num = #joint_names
    for i=1, joint_num do
--        print(gt_maps[rand_idx][i]:min(), gt_maps[rand_idx][i]:max())
--        print(outputs_map[rand_idx][i]:min(), outputs_map[rand_idx][i]:max())
        if gt_maps ~= nil then image.save(paths.concat(saveDir, string.format('frame_%s_gt.png', joint_names[i])), (255*gt_maps[rand_idx][i]):byte()) end
        image.save(paths.concat(saveDir, string.format('frame_%s_result.png', joint_names[i])), (255*outputs_map[rand_idx][i]):byte())
    end

    local compo_num = #compo_names
    for i=1, compo_num do
--        print(gt_maps[rand_idx][joint_num+2*i-1]:min(), gt_maps[rand_idx][joint_num+2*i-1]:max())
--        print(outputs_map[rand_idx][joint_num+2*i-1]:min(), outputs_map[rand_idx][joint_num+2*i-1]:max())
--        print(gt_maps[rand_idx][joint_num+2*i]:min(), gt_maps[rand_idx][joint_num+2*i]:max())
--        print(outputs_map[rand_idx][joint_num+2*i]:min(), outputs_map[rand_idx][joint_num+2*i]:max())

        if gt_maps ~= nil then image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gt.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+i]):byte()) end
        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_result.png', compo_names[i][1], compo_names[i][2])), (255*outputs_map[rand_idx][joint_num+i]):byte())


--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gtx.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+2*i-1]):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gty.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+2*i]):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resultx.png', compo_names[i][1],compo_names[i][2])), (255*outputs_map[rand_idx][joint_num+2*i-1]):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resulty.png', compo_names[i][1],compo_names[i][2])), (255*outputs_map[rand_idx][joint_num+2*i]):byte())

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
    self.toolCompoNames = opt.toolCompoNames

    self.toolJointNum = #self.toolJointNames
    self.toolCompoNum = #self.toolCompoNames


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
    self.mapsGPU = nil
    if nGPU > 1 then
        self.framesGPU = cutorch.createCudaHostTensor(self.batchSize, 3, self.inputHeight, self.inputWidth)
        self.mapsGPU = cutorch.createCudaHostTensor(self.batchSize, self.toolJointNum+2*self.toolCompoNum, self.inputHeight, self.inputWidth)
    else
--        self.framesGPU = torch.CudaTensor(self.batchSize, 3, self.inputHeight, self.inputWidth)
--        self.mapsGPU = torch.CudaTensor(self.batchSize, 1, self.inputHeight, self.inputWidth)
        self.framesGPU = torch.CudaTensor()
        self.mapsGPU = torch.CudaTensor()
    end

    self.params, self.gradParams = self.model:getParameters()
    print('model #params = ' .. tostring(#self.params))

    self.criterion = nn.BCECriterion()
--    self.criterion = nn.MSECriterion()
--    self.criterion.sizeAverage = false
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

    for n, framesCPU, mapsCPU in self.dataLoader:load(1) do
        -- load data
        dataTime = dataTime + dataTimer:time().real
        -- transfer over to GPU
        self.framesGPU:resize(framesCPU:size()):copy(framesCPU)
        self.mapsGPU:resize(mapsCPU:size()):copy(mapsCPU)

        -- reset gradparameters
        self.gradParams:zero()
        -- forward
--        print(self.framesGPU:size())
--        print(self.mapsGPU:size())
        local outputsGPU = self.model:forward(self.framesGPU)
        local loss_batch = self.criterion:forward(outputsGPU, self.mapsGPU)

        loss = loss + loss_batch
        -- backward
        local grad_output = self.criterion:backward(outputsGPU, self.mapsGPU)
        self.model:backward(self.framesGPU, grad_output)
        -- update parameters
        optim.sgd(feval, self.params, self.optimState)
        -- todo: accumulate accuracy
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.mapsGPU)):sum() / self.mapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == framesCPU:size(1) then
            print(mapsCPU:max())
            print(outputsGPU:max())
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames,'/home/xiaofei/workspace/toolPose/dual_results/train')
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
--    acc = 1 / (loss + 1e-5)
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

    for n, framesCPU, mapsCPU in self.dataLoader:load(2) do
        -- load data
        dataTime = dataTime + dataTimer:time().real
        -- transfer over to GPU
        self.framesGPU:resize(framesCPU:size()):copy(framesCPU)
        self.mapsGPU:resize(mapsCPU:size()):copy(mapsCPU)

        -- forward
--        print(self.framesGPU:size())
--        print(self.mapsGPU:size())
        local outputsGPU = self.model:forward(self.framesGPU)
        local loss_batch = self.criterion:forward(outputsGPU, self.mapsGPU)
        loss = loss + loss_batch
        -- todo: accumulate accuracy
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.mapsGPU)):sum() / self.mapsGPU:nElement()
        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == framesCPU:size(1) then
            print(mapsCPU:max())
            print(outputsGPU:max())
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames, '/home/xiaofei/workspace/toolPose/dual_results/val')
        end

        xlua.progress(n, size)
        collectgarbage()
        collectgarbage()
        dataTimer:reset()
    end

    -- calculate loss, acc
    loss = loss / N
    acc = acc * 100 / N
--    acc = 1 / (loss + 1e-5)
    print("\nVal : time to predict = " .. timer:time().real .. ' sec')
	print("Val : time to load data = " .. dataTime .. ' sec')
    return acc, loss
end

function Runner:test(epoch)
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local dataTime = 0
    local size = self.dataLoader:testSize()
    local loss = 0.0
    local acc = 0.0
    local N = 0

    self.model:evaluate()

    for n, framesCPU, mapsCPU in self.dataLoader:load(3) do
        -- load data
        dataTime = dataTime + dataTimer:time().real
        -- transfer over to GPU
        self.framesGPU:resize(framesCPU:size()):copy(framesCPU)

        -- forward
--        print(self.framesGPU:size())
        local outputsGPU = self.model:forward(self.framesGPU)
--        local loss_batch = self.criterion:forward(outputsGPU, self.mapsGPU)
--        loss = loss + loss_batch
        --  accumulate accuracy
--        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.mapsGPU)):sum() / self.mapsGPU:nElement()
--        acc = acc + batch_acc
        N = N + 1

        -- visualize result for debugging
        if n == framesCPU:size(1) then
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames, '/home/xiaofei/workspace/toolPose/dual_results/test')
        end

        xlua.progress(n, size)
        collectgarbage()
        collectgarbage()
        dataTimer:reset()
    end

    -- calculate loss, acc
    loss = loss / N
    acc = acc * 100 / N
--    acc = 1 / (loss + 1e-5)
    print("\nTest : time to predict = " .. timer:time().real .. ' sec')
	print("Test : time to load data = " .. dataTime .. ' sec')
    return acc, loss

end

return M.Runner

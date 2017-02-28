require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'xlua'
require 'image'
local optim = require('optim')
local matio = require 'matio'
local DataLoader = require 'dataloader_invivo_det'

torch.setdefaulttensortype('torch.FloatTensor')

local ORIGIN_FRAME_HEIGHT = 1080 --576
local ORIGIN_FRAME_WIDTH = 1920 --720

-- frames       [b,c=3,H,W]
-- gt_jtmaps    [b,c=1,h,w]
-- outputs      [b*h*w,9]
local function visualResult(frames, gt_maps, outputs_map, joint_names, compo_names, saveDir)
--    outputs_map:clamp(0,1)
    local batch_size = frames:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), (255*frames[rand_idx]):byte())

    local joint_num = #joint_names
    for i=1, joint_num do
--        print(gt_maps[rand_idx][i]:min(), gt_maps[rand_idx][i]:max())
--        print(outputs_map[rand_idx][i]:min(), outputs_map[rand_idx][i]:max())
        if gt_maps ~= nil then
            image.save(paths.concat(saveDir, string.format('frame_%s_gt.png', joint_names[i])), (255*gt_maps[rand_idx][i]):byte())
        end
        image.save(paths.concat(saveDir, string.format('frame_%s_result.png', joint_names[i])), (255*outputs_map[rand_idx][i]):byte())
    end

    local compo_num = #compo_names
    for i=1, compo_num do
--        print(gt_maps[rand_idx][joint_num+i]:min(), gt_maps[rand_idx][joint_num+i]:max())
--        print(outputs_map[rand_idx][joint_num+i]:min(), outputs_map[rand_idx][joint_num+i]:max())

        if gt_maps ~= nil then
            image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gt.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+i]):byte())
        end
        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_result.png', compo_names[i][1], compo_names[i][2])), (255*outputs_map[rand_idx][joint_num+i]):byte())
    end
end

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

local function JointPrecision(gt_joints_anno, outputs_map, joint_names, dist_thres)
    local batch_size = outputs_map:size(1)
    local output_height = outputs_map:size(3)
    local output_width = outputs_map:size(4)
    local batch_output_peaks_tab = {}
    dist_thres = dist_thres or 10
    local score_factor_thres = 0.3

    -- nms
    for bidx = 1, batch_size do
        local output_peaks_tab = {}
        for i=1, #joint_names do
            local joint_map = outputs_map[bidx][i]
            local output_peaks = nmsPt(joint_map, dist_thres, score_factor_thres)
            for pidx=1, #output_peaks do
--                print(output_peaks[pidx][1], output_peaks[pidx][2])
                output_peaks[pidx][1] = output_peaks[pidx][1] / output_height * ORIGIN_FRAME_HEIGHT
                output_peaks[pidx][2] = output_peaks[pidx][2] / output_width * ORIGIN_FRAME_WIDTH
            end
            table.insert(output_peaks_tab, output_peaks)
        end
        table.insert(batch_output_peaks_tab, output_peaks_tab)
    end

    local dist_tab
    local old_dist_tab = {}
    local new_dist_tab = {}
    for i=1, #joint_names do
        table.insert(old_dist_tab, {})
        table.insert(new_dist_tab, {})
    end

    for bidx=1, batch_size do
        local frame_anno = gt_joints_anno[bidx].anno
        local frame_class = gt_joints_anno[bidx].class

        if frame_class == 1 then
            dist_tab = old_dist_tab
        elseif frame_class == 2 then
            dist_tab = new_dist_tab
        end

        for i=1, #joint_names do
            local joint_anno = frame_anno[joint_names[i]]
            if joint_anno ~= nil then
                local output_peaks = batch_output_peaks_tab[bidx][i]
                for tool_idx=1, #joint_anno do
                    local gt_joint_x = joint_anno[tool_idx].x * ORIGIN_FRAME_WIDTH
                    local gt_joint_y = joint_anno[tool_idx].y * ORIGIN_FRAME_HEIGHT
--                    print(string.format('gtx=%f, gty=%f', gt_joint_x, gt_joint_y))
                    -- compute the distance
                    local dist = 1e+8
                    local chosen_result_x, chosen_result_y = -1, -1
                    for pidx=1, #output_peaks do
                        local result_joint_x = output_peaks[pidx][2]
                        local result_joint_y = output_peaks[pidx][1]
--                        print(string.format('rex=%f, rey=%f', result_joint_x, result_joint_y))
                        local d = math.sqrt(math.pow(gt_joint_x-result_joint_x,2)+math.pow(gt_joint_y-result_joint_y,2))
                        if d < dist then
                            dist = d
                            chosen_result_x = result_joint_x
                            chosen_result_y = result_joint_y
                        end
                    end
--                    if dist > 20 and dist < 1e+8 then
--                        print('')
--                        print(joint_names[i])
--                        for pidx=1, #output_peaks do
--                            print(string.format('candis:[%d, %d]', output_peaks[pidx][2], output_peaks[pidx][1]))
--                        end
--                        print(string.format('gt:    [%d, %d]', gt_joint_x, gt_joint_y))
--                        print(string.format('result:[%d, %d]', chosen_result_x, chosen_result_y))
--                        print(string.format('dist = %.2f', dist))
--                    end
                    table.insert(dist_tab[i], dist)
                end
            end
        end
    end

    -- average dist
    local old_avg_dist_tab = {}
    for i=1, #old_dist_tab do
        local avg_dist = 0.0
        local joint_disttab = old_dist_tab[i]
        for j=1, #joint_disttab do
            avg_dist = avg_dist + joint_disttab[j]
        end
        if #joint_disttab ~= 0 then avg_dist = avg_dist / #joint_disttab end
        table.insert(old_avg_dist_tab, avg_dist)
--        print(string.format('%s average precision dist = %.2f',joint_names[i], avg_dist))
    end

    local new_avg_dist_tab = {}
    for i=1, #new_dist_tab do
        local avg_dist = 0.0
        local joint_disttab = new_dist_tab[i]
        for j=1, #joint_disttab do
            avg_dist = avg_dist + joint_disttab[j]
        end
        if #joint_disttab ~= 0 then avg_dist = avg_dist / #joint_disttab end
        table.insert(new_avg_dist_tab, avg_dist)
--        print(string.format('%s average precision dist = %.2f',joint_names[i], avg_dist))
    end

--    local avgdist = 0.0
--    for i=1, #avg_dist_tab do
--        avgdist = avgdist + avg_dist_tab[i]
--    end
--    avgdist = avgdist / #avg_dist_tab
--    return avg_dist_tab
    return old_avg_dist_tab, new_avg_dist_tab
end

local M = {}
local Runner = torch.class('Runner', M)
function Runner:__init(net_path, opt, optimState)
    -- load network
    print('Loading network ...')
    self.model = torch.load(net_path)
--    print(self.model)
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

    self.jointRadius = opt.jointRadius or 10


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
    local old_acc = 0.0
    local new_acc = 0.0
    local old_prec_tab = {}
    local new_prec_tab = {}
    for i=1, #self.toolJointNames do
        table.insert(old_prec_tab, 0.0)
        table.insert(new_prec_tab, 0.0)
    end
    local N = 0

    self.model:training()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    for n, framesCPU, mapsCPU, jointAnnoTab in self.dataLoader:load(1) do
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
        local old_batch_prec, new_batch_prec = {}, {}
        for i=1, #self.toolJointNames do table.insert(old_batch_prec, -1) table.insert(new_batch_prec, -1) end
        old_batch_prec, new_batch_prec = JointPrecision(jointAnnoTab, outputsGPU, self.toolJointNames, self.jointRadius)
        for i=1, #self.toolJointNames do
            old_prec_tab[i] = old_batch_prec[i] + old_batch_prec[i]
            new_prec_tab[i] = new_prec_tab[i] + new_batch_prec[i]
        end
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.mapsGPU)):sum() / self.mapsGPU:nElement()
        acc = acc + batch_acc

        -- acc for old or new data
        local old_batch_acc, new_batch_acc = 0.0, 0.0
        local old_num, new_num = 0, 0
        for idx = 1, #jointAnnoTab do
            if jointAnnoTab[idx].class == 1 then -- old
                old_batch_acc = old_batch_acc + torch.eq(torch.round(outputsGPU[idx]), torch.round(self.mapsGPU[idx])):sum() / self.mapsGPU[idx]:nElement()
                old_num = old_num + 1
            else
                new_batch_acc = new_batch_acc + torch.eq(torch.round(outputsGPU[idx]), torch.round(self.mapsGPU[idx])):sum() / self.mapsGPU[idx]:nElement()
                new_num = new_num + 1
            end
        end
        old_acc = old_acc + old_batch_acc/old_num
        new_acc = new_acc + new_batch_acc/new_num

        N = N + 1

        -- visualize result for debugging
        if n == framesCPU:size(1) then
            print(mapsCPU:max())
            print(outputsGPU:max())
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames,'/home/xiaofei/workspace/toolPose/finetune_det_results/train')
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
    old_acc = old_acc * 100 / N
    new_acc = new_acc * 100 / N

    local old_prec = 0.0    -- average joint prec
    for i=1, #self.toolJointNames do
        old_prec_tab[i] = old_prec_tab[i] / N
--        print(string.format('OLD %s average precision dist = %.2f',self.toolJointNames[i], old_prec_tab[i]))
        old_prec = old_prec + old_prec_tab[i]
    end
    old_prec = old_prec / #self.toolJointNames

    local new_prec = 0.0 -- average joint prec
    for i=1, #self.toolJointNames do
        new_prec_tab[i] = new_prec_tab[i] / N
--        print(string.format('NEW %s average precision dist = %.2f',self.toolJointNames[i], new_prec_tab[i]))
        new_prec = new_prec + new_prec_tab[i]
    end
    new_prec = new_prec / #self.toolJointNames

--    acc = 1 / (loss + 1e-5)
    print("\nTrain : time to learn = " .. timer:time().real .. ' sec')
	print("Train : time to load data = " .. dataTime .. ' sec')
    return acc, loss, old_acc, new_acc
end

function Runner:val(epoch)
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local dataTime = 0
    local size = self.dataLoader:valSize()
    local loss = 0.0
    local acc = 0.0
    local old_acc = 0.0
    local new_acc = 0.0
    local old_prec_tab = {}
    local new_prec_tab = {}
    for i=1, #self.toolJointNames do
        table.insert(old_prec_tab, 0.0)
        table.insert(new_prec_tab, 0.0)
    end
    local N = 0

    self.model:evaluate()

    for n, framesCPU, mapsCPU, jointAnnoTab in self.dataLoader:load(2) do
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
        local old_batch_prec, new_batch_prec = {}, {}
        for i=1, #self.toolJointNames do table.insert(old_batch_prec, -1) table.insert(new_prec_tab, -1) end
        old_batch_prec, new_batch_prec = JointPrecision(jointAnnoTab, outputsGPU, self.toolJointNames, self.jointRadius)
        for i=1, #self.toolJointNames do
            old_prec_tab[i] = old_prec_tab[i] + old_batch_prec[i]
            new_prec_tab[i] = new_prec_tab[i] + new_batch_prec[i]
        end
        local batch_acc = torch.eq(torch.round(outputsGPU), torch.round(self.mapsGPU)):sum() / self.mapsGPU:nElement()
        acc = acc + batch_acc

        -- acc for old or new data
        local old_batch_acc, new_batch_acc = 0.0, 0.0
        local old_num, new_num = 0, 0
        for idx = 1, #jointAnnoTab do
            if jointAnnoTab[idx].class == 1 then -- old
                old_batch_acc = old_batch_acc + torch.eq(torch.round(outputsGPU[idx]), torch.round(self.mapsGPU[idx])):sum() / self.mapsGPU[idx]:nElement()
                old_num = old_num + 1
            else
                new_batch_acc = new_batch_acc + torch.eq(torch.round(outputsGPU[idx]), torch.round(self.mapsGPU[idx])):sum() / self.mapsGPU[idx]:nElement()
                new_num = new_num + 1
            end
        end
        old_acc = old_acc + old_batch_acc/old_num
        new_acc = new_acc + new_batch_acc/new_num

        N = N + 1

        -- visualize result for debugging
        if n == framesCPU:size(1) then
            print(mapsCPU:max())
            print(outputsGPU:max())
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames, '/home/xiaofei/workspace/toolPose/finetune_det_results/val')
        end

        xlua.progress(n, size)
        collectgarbage()
        collectgarbage()
        dataTimer:reset()
    end

    -- calculate loss, acc
    loss = loss / N
    acc = acc * 100 / N
    old_acc = old_acc * 100 / N
    new_acc = new_acc * 100 / N

    local old_prec = 0.0    -- average joint prec
    for i=1, #self.toolJointNames do
        old_prec_tab[i] = old_prec_tab[i] / N
--        print(string.format('OLD %s average precision dist = %.2f',self.toolJointNames[i], old_prec_tab[i]))
        old_prec = old_prec + old_prec_tab[i]
    end
    old_prec = old_prec / #self.toolJointNames

    local new_prec = 0.0 -- average joint prec
    for i=1, #self.toolJointNames do
        new_prec_tab[i] = new_prec_tab[i] / N
--        print(string.format('NEW %s average precision dist = %.2f',self.toolJointNames[i], new_prec_tab[i]))
        new_prec = new_prec + new_prec_tab[i]
    end
    new_prec = new_prec / #self.toolJointNames

--    acc = 1 / (loss + 1e-5)
    print("\nVal : time to predict = " .. timer:time().real .. ' sec')
	print("Val : time to load data = " .. dataTime .. ' sec')
    return acc, loss, old_acc, new_acc
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
            saveMatResult(framesCPU, mapsCPU, outputsGPU, self.toolJointNames, self.toolCompoNames, '/home/xiaofei/workspace/toolPose/finetune_det_results/test')
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

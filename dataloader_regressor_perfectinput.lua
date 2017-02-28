-- the input of the regression network are perfect output of the detection network

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
require 'data_utils_new'
torch.setdefaulttensortype('torch.FloatTensor')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader:__init(opt)
    self.opt = opt
    self.trainStyle = opt.trainStyle
    self.trainBatchSize = opt.trainBatchSize or opt.batchSize or 1
    self.valBatchSize = opt.valBatchSize or opt.batchSize or 1
    self.testBatchSize = opt.testBatchSize or opt.batchSize or 1
    -- get data
    local train_data_tab, val_data_tab, test_data_tab
    local train_data_file = paths.concat(opt.dataDir, 'train_endo_toolpos_head.t7')
    if paths.filep(train_data_file) then
        train_data_tab = torch.load(train_data_file)
    else
        error('no such file: ' .. train_data_file)
    end
    local val_data_file = paths.concat(opt.dataDir, 'val_endo_toolpos_head.t7')
    if paths.filep(val_data_file) then
        val_data_tab = torch.load(val_data_file)
    else
        error('no such file: ' .. val_data_file)
    end
    local test_data_file = paths.concat(opt.dataDir, 'test_endo_frames.t7')
    if paths.filep(test_data_file) then
        test_data_tab = torch.load(test_data_file)
    else
        error('no such file: ' .. test_data_file)
    end
    self.testDataTab = test_data_tab

    self.rotMaxDegree = opt.rotMaxDegree or 20
    -- rotation and flip to augment data
    self.vflip = opt.vflip or 0
    self.hflip = opt.hflip or 0

    -- construct aug data and the rotation and flip parameter table for training data
    self.augTrainParamTab = {}
    self.trainDataTab = {}
    for i=1, #train_data_tab do
        for deg = -1 * self.rotMaxDegree, self.rotMaxDegree do
            for vflip = 0, self.vflip do
                for hflip = 0, self.hflip do
                    table.insert(self.trainDataTab, train_data_tab[i])
                    table.insert(self.augTrainParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end
    self.augValParamTab = {}
    self.valDataTab = {}
    for i=1, #val_data_tab do
        for deg = -1 * 0, 0 do
            for vflip = 0, 0 do
                for hflip = 0, 0 do
                    table.insert(self.valDataTab, val_data_tab[i])
                    table.insert(self.augValParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end

    print(string.format('Generate %d train Samples', #train_data_tab))
    print(string.format('Augmented into %d samples', #self.trainDataTab))
    print(string.format('Generate %d val Samples' , #val_data_tab))
    print(string.format('Augmented into %d samples', #self.valDataTab))

    -- dataloader
    self.trainBatches = math.floor(#self.trainDataTab / self.trainBatchSize)
    self.trainSamples = self.trainBatches * self.trainBatchSize
    self.valBatches = math.floor(#self.valDataTab / self.valBatchSize)
    self.valSamples = self.valBatches * self.valBatchSize
    self.testBatches = math.floor(#self.testDataTab / self.testBatchSize)
    self.testSamples = self.testBatches * self.testBatchSize
    print('Train Sample number: ' .. self.trainSamples)
    print('Val Sample number: ' .. self.valSamples)
    print('Test Sample number: ' .. self.testSamples)
    print('==================================================================')
    self.pool = Threads(
        opt.nThreads,
        function(thread_id)
                require('torch')
                require('image')
                require('data_utils_new')
        end,
        function(thread_id)
                torch.setdefaulttensortype('torch.FloatTensor')
        end
    )

--    self.mean = torch.Tensor({123, 117, 104}) -- NOTE: RGB style
    self.inputWidth = opt.inputWidth or 720
    self.inputHeight = opt.inputHeight or 576
    self.toolJointNames = opt.toolJointNames
    self.toolCompoNames = opt.toolCompoNames
    self.jointRadius = opt.jointRadius or 20
    self.detJointRadius = opt.detJointRadius or 10
    self.modelOutputScale = opt.modelOutputScale or 4
    self.normalScale = opt.normalScale or 1
    print(string.format('Det Model radius = %f', self.detJointRadius))
    print(string.format('Regression Model radius = %f', self.jointRadius))
    print(string.format('Regression normalize scale = %f', self.normalScale))
end

function DataLoader:trainSize()
    return self.trainSamples
end

function DataLoader:valSize()
    return self.valSamples
end

function DataLoader:testSize()
--    return self.testSamples
    return self.testBatchSize
end

function DataLoader:load(job_type)
    local batch_size, nSamples, data_tab, aug_param_tab
    if job_type == 1 then
        batch_size = self.trainBatchSize
        data_tab = self.trainDataTab
        nSamples = self.trainSamples
        aug_param_tab = self.augTrainParamTab
    elseif job_type == 2 then
        batch_size = self.valBatchSize
        data_tab = self.valDataTab
        nSamples = self.valSamples
        aug_param_tab = self.augValParamTab
    else
        batch_size = self.testBatchSize
        data_tab = self.testDataTab
        nSamples = self.testBatchSize
        aug_param_tab = nil
    end
    local perm = torch.randperm(#data_tab)
    local pool = self.pool
    local input_width = self.inputWidth
    local input_height = self.inputHeight

    local jointNum = #self.toolJointNames
    local compoNum = #self.toolCompoNames

    local jointNames = self.toolJointNames
    local compoNames = self.toolCompoNames

    local j_radius = self.jointRadius
    local det_j_radius = self.detJointRadius
    local model_output_scale = self.modelOutputScale
    local normal_scale = self.normalScale

    local train_style = self.trainStyle

    local job_done = 0
    local idx = 1
    local frame_batch_CPU, frame_batch_det_map_CPU, frame_batch_map_CPU, frame_batch_anno_CPU
    local function enqueue()
        while idx <= nSamples and pool:acceptsjob() do
            local _indices = perm:narrow(1, idx, batch_size)
            idx = idx + batch_size
            pool:addjob(
            function(indices)
                -- load data
                local frame_batch_map, frame_batch_det_map
                local joint_batch_anno = {}
                local frame_tab = {}
                if job_type ~= 1 and job_type ~= 2 then
                    frame_batch_map = nil
                    frame_batch_det_map = nil

                    for i=1, batch_size do
                        local frame_data = data_tab[indices[i]]
                        local frame = image.load(frame_data.filename, 3, 'byte')
                        frame = image.scale(frame, input_width, input_height)
                        table.insert(frame_tab, frame)
                    end
                else
                    if train_style == nil then
                        frame_batch_det_map = torch.FloatTensor(batch_size, jointNum+compoNum,
                                                                torch.floor(input_height/model_output_scale),
                                                                torch.floor(input_width/model_output_scale))
                    end
                    frame_batch_map = torch.FloatTensor(batch_size, jointNum+compoNum,
                                                        torch.floor(input_height/model_output_scale),
                                                        torch.floor(input_width/model_output_scale))
                    for i=1, batch_size do
                        local frame_data = data_tab[indices[i]]
                        local aug_param = aug_param_tab[indices[i]]
                        local frame = image.load(frame_data.filename, 3, 'byte')
                        frame = image.scale(frame, input_width, input_height)
                        -- augment data
                        local aug_frame, aug_annos
                        aug_frame, aug_annos = flipToolPosData(frame,  aug_param.hflip, aug_param.vflip, frame_data.annotations)
--                        aug_frame, aug_annos = rotateToolPos(aug_frame, aug_param.degree, aug_annos)
                        table.insert(frame_tab, aug_frame)
                        -- note: joint map size is based on the model output: scale down 1 or 4 times

                        if train_style == nil then
                            local jointdetmap = genSepJointMap(aug_annos, jointNames, det_j_radius, aug_frame, model_output_scale)
                            frame_batch_det_map[{i, {1, jointNum}}] = jointdetmap:clone()
                            local compodetmap = genSepPAFMapDet(aug_annos, compoNames, det_j_radius, aug_frame, model_output_scale)
                            frame_batch_det_map[{i, {jointNum+1, -1}}] = compodetmap:clone()
                        end

                        local heatmap = genSepHeatMap(aug_annos, jointNames, j_radius, det_j_radius, aug_frame, model_output_scale, normal_scale)
                        frame_batch_map[{i, {1, jointNum}}] = heatmap:clone()
                        -- todo: paf heatmap generation
                        local compmap = genSepPAFMapReg(aug_annos, compoNames, j_radius, det_j_radius, aug_frame, model_output_scale, normal_scale)
                        frame_batch_map[{i, {jointNum+1, -1}}] = compmap:clone()

--                        local joint_norm_pos = getJointPos(aug_annos, jointNames, frame_data.toolNum)
--                        joint_batch_pos[i] = joint_norm_pos:clone()
                        table.insert(joint_batch_anno, aug_annos)
                    end
                end
                -- preprocess images
                local frame_batch = preProcess(frame_tab, input_width, input_height)
                collectgarbage()
                collectgarbage()
                return frame_batch, frame_batch_det_map, frame_batch_map, joint_batch_anno
            end,
            function(frame_batch, frame_batch_det_map, frame_batch_map, joint_batch_anno)
                frame_batch_CPU = frame_batch
                frame_batch_det_map_CPU = frame_batch_det_map
                frame_batch_map_CPU = frame_batch_map
                frame_batch_anno_CPU = joint_batch_anno
                job_done = job_done + batch_size
            end,
            _indices
            )
        end
    end
    local function loop()
        enqueue()
        if not pool:hasjob() then
            return nil
        end
        pool:dojob()
        if pool:haserror() then
            pool:synchronize()
        end
        enqueue()
        return job_done, frame_batch_CPU,  frame_batch_det_map_CPU, frame_batch_map_CPU, frame_batch_anno_CPU
    end
    return loop
end

function DataLoader:terminate()
    self.pool:terminate()
end

return M.DataLoader
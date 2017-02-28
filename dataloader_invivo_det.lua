local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
torch.setdefaulttensortype('torch.FloatTensor')

require 'data_utils_new'

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader:__init(opt)
    self.opt = opt
    self.trainBatchSize = opt.trainBatchSize or opt.batchSize or 1
    self.valBatchSize = opt.valBatchSize or opt.batchSize or 1
    self.testBatchSize = opt.testBatchSize or opt.batchSize or 1

    -- get original data
    local old_train_data_tab, old_val_data_tab
    local train_data_file = paths.concat(opt.oldDataDir, 'train_endo_toolpos_head.t7')
    if paths.filep(train_data_file) then old_train_data_tab = torch.load(train_data_file) else error('no such file: ' .. train_data_file) end
    local val_data_file = paths.concat(opt.oldDataDir, 'val_endo_toolpos_head.t7')
    if paths.filep(val_data_file) then old_val_data_tab = torch.load(val_data_file) else error('no such file: ' .. val_data_file) end

    -- train with the whole training dataset
    for i=1, #old_val_data_tab do table.insert(old_train_data_tab, old_val_data_tab[i]) end
    old_val_data_tab = nil
    old_val_data_tab = old_train_data_tab


    -- get finetune data
    self.dataType = opt.dataType or 'invivo'
    local new_train_data_tab, new_val_data_tab
    train_data_file = paths.concat(opt.newDataDir, string.format('train_%s_toolpos.t7', self.dataType))
    if paths.filep(train_data_file) then new_train_data_tab = torch.load(train_data_file) else error('no such file: ' .. train_data_file) end
    val_data_file = paths.concat(opt.newDataDir, string.format('val_%s_toolpos.t7', self.dataType))
    if paths.filep(val_data_file) then new_val_data_tab = torch.load(val_data_file) else error('no such file: ' .. val_data_file) end

    local test_data_tab
    local test_data_file = paths.concat(opt.newDataDir, string.format('test_%s_frames.t7', self.dataType))
    if paths.filep(test_data_file) then test_data_tab = torch.load(test_data_file) else error('no such file: ' .. test_data_file) end
    self.testDataTab = test_data_tab

    -- flip to augment data
    self.vflip = opt.vflip or 0
    self.hflip = opt.hflip or 0

    -- construct aug data and the flip parameter table for training data
    self.oldAugTrainParamTab = {}
    self.oldTrainDataTab = {}
    for i=1, #old_train_data_tab do
        for deg = -1*0, 0 do
            for vflip = 0, self.vflip do
                for hflip = 0, self.hflip do
                    table.insert(self.oldTrainDataTab, old_train_data_tab[i])
                    table.insert(self.oldAugTrainParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end
    self.oldAugValParamTab = {}
    self.oldValDataTab = {}
    for i=1, #old_val_data_tab do
        for deg = -1 * 0, 0 do
            for vflip = 0, 0 do
                for hflip = 0, 0 do
                    table.insert(self.oldValDataTab, old_val_data_tab[i])
                    table.insert(self.oldAugValParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end

    print(string.format('OLD: Generate %d train Samples', #old_train_data_tab))
    print(string.format('OLD: Augmented into %d samples', #self.oldTrainDataTab))
    print(string.format('OLD: Generate %d val Samples' , #old_val_data_tab))
    print(string.format('OLD: Augmented into %d samples', #self.oldValDataTab))

    self.newAugTrainParamTab = {}
    self.newTrainDataTab = {}
    for i=1, #new_train_data_tab do
        for deg = -1*0, 0 do
            for vflip = 0, self.vflip do
                for hflip = 0, self.hflip do
                    table.insert(self.newTrainDataTab, new_train_data_tab[i])
                    table.insert(self.newAugTrainParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end

    self.newAugValParamTab = {}
    self.newValDataTab = {}
    for i=1, #new_train_data_tab do
        for deg = -1*0, 0 do
            for vflip = 0, 0 do
                for hflip = 0, 0 do
                    table.insert(self.newValDataTab, new_val_data_tab[i])
                    table.insert(self.newAugValParamTab, {degree=deg, vflip=vflip, hflip=hflip})
                end
            end
        end
    end

    print(string.format('NEW: Generate %d train Samples', #new_train_data_tab))
    print(string.format('NEW: Augmented into %d samples', #self.newTrainDataTab))
    print(string.format('NEW: Generate %d val Samples' , #new_val_data_tab))
    print(string.format('NEW: Augmented into %d samples', #self.newValDataTab))
    -- dataloader
    local minTrainDataSamples = #self.newTrainDataTab * 2
    local minValDataSamples = #self.newValDataTab * 2
    self.trainBatches = math.floor(minTrainDataSamples / self.trainBatchSize)
    self.trainSamples = self.trainBatches * self.trainBatchSize
    self.valBatches = math.floor(minValDataSamples / self.valBatchSize)
    self.valSamples = self.valBatches * self.valBatchSize
    self.testBatches = math.floor(#self.testDataTab / self.testBatchSize)
    self.testSamples = self.testBatches * self.testBatchSize
    print('Train Sample number: ' .. self.trainSamples)
    print('Val Sample number" ' .. self.valSamples)
    print('Test Sample number: ' .. self.testSamples)
    print('================================================================')
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
    local batch_size, nSamples
    local old_data_tab, new_data_tab, data_tab
    local old_aug_param_tab, new_aug_param_tab, aug_param_tab
    if job_type == 1 then
        batch_size = self.trainBatchSize
        nSamples = self.trainSamples
        old_data_tab = self.oldTrainDataTab
        new_data_tab = self.newTrainDataTab
        old_aug_param_tab = self.oldAugTrainParamTab
        new_aug_param_tab = self.newAugTrainParamTab
    elseif job_type == 2 then
        batch_size = self.valBatchSize
        nSamples = self.valSamples
        old_data_tab = self.oldValDataTab
        new_data_tab = self.newValDataTab
        old_aug_param_tab = self.oldAugValParamTab
        new_aug_param_tab = self.newAugValParamTab
    else
        batch_size = self.testBatchSize
        nSamples = self.testBatchSize * 2
        old_data_tab = self.testDataTab
        new_data_tab = self.testDataTab
        old_aug_param_tab = nil
        new_aug_param_tab = nil
    end
    data_tab = {old_data_tab, new_data_tab }
    aug_param_tab = {old_aug_param_tab, new_aug_param_tab }

    local new_perm = torch.randperm(#new_data_tab)
    local old_pm = torch.randperm(#old_data_tab)
    local old_perm = torch.IntTensor(#new_data_tab)
    for r=1, old_perm:size(1) do
        local idx = r % #old_data_tab
        if idx == 0 then idx = #old_data_tab end
        old_perm[r] = old_pm[idx]
    end
    local perm = torch.IntTensor(nSamples, 2)
    for i=1, nSamples/2 do
        perm[i*2-1][1] = 1 -- old
        perm[i*2-1][2] = old_perm[i]
        perm[i*2][1] = 2 -- new
        perm[i*2][2] = new_perm[i]
    end


    local pool = self.pool
    local input_width = self.inputWidth
    local input_height = self.inputHeight

    local jointNum = #self.toolJointNames
    local compoNum = #self.toolCompoNames

    local jointNames = self.toolJointNames
    local compoNames = self.toolCompoNames

    local j_radius = self.jointRadius
    local model_output_scale = self.modelOutputScale

    local job_done = 0
    local idx = 1
    local frame_batch_CPU, frame_batch_map_CPU, frame_batch_anno_CPU
    local function enqueue()
        while idx <= nSamples and pool:acceptsjob() do
            local _indices = perm:narrow(1, idx, batch_size)
            idx = idx + batch_size
            pool:addjob(
            function(indices)
                -- load data
                local frame_batch_map
                local joint_batch_anno = {}
                local frame_tab = {}
                if job_type ~= 1 and job_type ~= 2 then
                    frame_batch_map = nil
                    for i=1, batch_size do
                        local data_type = indices[i][1]
                        local data_idx = indices[i][2]
                        local frame_data = data_tab[data_type][data_idx]
                        local frame = image.load(frame_data.filename, 3, 'byte')
                        frame = image.scale(frame, input_width, input_height)
                        table.insert(frame_tab, frame)
                    end
                else
                    frame_batch_map = torch.FloatTensor(batch_size, jointNum+compoNum,
                                                        torch.floor(input_height/model_output_scale),
                                                        torch.floor(input_width/model_output_scale))
                    for i=1, batch_size do
                        local data_type = indices[i][1]
                        local data_idx = indices[i][2]
                        local frame_data = data_tab[data_type][data_idx]
                        local aug_param = aug_param_tab[data_type][data_idx]
                        local frame = image.load(frame_data.filename, 3, 'byte')
                        frame = image.scale(frame, input_width, input_height)
                        -- augment data
                        local aug_frame, aug_annos
                        aug_frame, aug_annos = flipToolPosData(frame, aug_param.hflip, aug_param.vflip, frame_data.annotations)
--                            aug_frame, aug_annos = rotateToolPos(aug_frame, aug_param.degree, aug_annos)
                        table.insert(frame_tab, aug_frame)
                        -- joint map generation
                        -- note: joint map size is based on the model output: scale down 4 times
--                        local heatmap = genSepHeatMap(aug_annos, jointNames, j_radius, aug_frame, model_output_scale)
                        local jointmap = genSepJointMap(aug_annos, jointNames, j_radius, aug_frame, model_output_scale)
                        frame_batch_map[{i, {1, jointNum}}] = jointmap:clone()
--                        local compmap = genSepPAFMap(aug_annos, compoNames, j_radius, aug_frame, model_output_scale)
                        local compmap = genSepPAFMapDet(aug_annos, compoNames, j_radius, aug_frame, model_output_scale)
                        frame_batch_map[{i, {jointNum+1, -1}}] = compmap:clone()
                        table.insert(joint_batch_anno, {class=data_type, anno=aug_annos})
                    end
                end
                -- preprocess images
                local frame_batch = preProcess(frame_tab, input_width, input_height)
                collectgarbage()
                collectgarbage()
                return frame_batch, frame_batch_map, joint_batch_anno
            end,
            function(frame_batch, frame_batch_map, joint_batch_anno)
                frame_batch_CPU = frame_batch
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
        if not pool:hasjob() then return nil end
        pool:dojob()
        if pool:haserror() then pool:synchronize() end
        enqueue()
        return job_done, frame_batch_CPU, frame_batch_map_CPU, frame_batch_anno_CPU
    end
    return loop
end

function DataLoader:terminate()
    self.pool:terminate()
end

return M.DataLoader











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

    -- get data
    local train_data_tab, val_data_tab
    local train_data_file = paths.concat(opt.dataDir, 'train_endo_toolpos.t7')
    if paths.filep(train_data_file) then
        train_data_tab = torch.load(train_data_file)
    else
        error('no such file: ' .. train_data_file)
    end
    local val_data_file = paths.concat(opt.dataDir, 'val_endo_toolpos.t7')
    if paths.filep(val_data_file) then
        val_data_tab = torch.load(val_data_file)
    else
        error('no such file: ' .. val_data_file)
    end

    self.rotMaxDegree = opt.rotMaxDegree or 20
    -- rotation and flip to augment data
    self.flipFlag = opt.flipFlag or 0

    -- construct aug data and the rotation and flip parameter table for training data
    self.augTrainParamTab = {}
    self.trainDataTab = {}
    for i=1, #train_data_tab do
        for deg = -1 * self.rotMaxDegree, self.rotMaxDegree do
            for flip = 0, self.flipFlag do
                table.insert(self.trainDataTab, train_data_tab[i])
                table.insert(self.augTrainParamTab, {degree=deg, flip=flip})
            end
        end
    end
    self.augValParamTab = {}
    self.valDataTab = {}
    for i=1, #val_data_tab do
        for deg = -1 * 0, 0 do
            for flip = 0, self.flipFlag do
                table.insert(self.valDataTab, val_data_tab[i])
                table.insert(self.augValParamTab, {degree=deg, flip=flip})
            end
        end
    end

    print(string.format('Generate %d train Samples', #train_data_tab))
    print(string.format('For each train sample, augmented into %d samples', #self.augTrainParamTab))
    print(string.format('Generate %d val Samples' , #val_data_tab))
    print(string.format('For each val sample, augmented into %d samples', #self.valDataTab))

    -- dataloader
    self.trainBatches = math.floor(#self.trainDataTab / self.trainBatchSize)
    self.trainSamples = self.trainBatches * self.trainBatchSize
    self.valBatches = math.floor(#self.valDataTab / self.valBatchSize)
    self.valSamples = self.valBatches * self.valBatchSize
    print('Train Sample number: ' .. self.trainSamples)
    print('Val Sample number: ' .. self.valSamples)
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

    self.mean = torch.Tensor({123, 117, 104}) -- NOTE: RGB style
    self.inputWidth = opt.inputWidth or 720
    self.inputHeight = opt.inputHeight or 576
    self.toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint',
                          'TrackedPoint', 'EndPoint' } -- joint number = 6
    self.jointRadius = opt.jointRadius or 20
    self.modelOutputScale = opt.modelOutputScale or 4
end

function DataLoader:trainSize()
    return self.trainSamples
end

function DataLoader:valSize()
    return self.valSamples
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
    end
    local perm = torch.randperm(#data_tab)
    local pool = self.pool
    local input_width = self.inputWidth
    local input_height = self.inputHeight
    local mean_ = self.mean
    local jointNames = self.toolJointNames

    local j_radius = self.jointRadius
    local model_output_scale = self.modelOutputScale

    local job_done = 0
    local idx = 1
    local frame_batch_CPU, frame_batch_jointmap_CPU
    local function enqueue()
        while idx <= nSamples and pool:acceptsjob() do
            local _indices = perm:narrow(1, idx, batch_size)
            idx = idx + batch_size
            pool:addjob(
                function(indices)
                    -- load data
                    local frame_batch_jointmap = torch.FloatTensor(batch_size, 1,
                                                                   torch.floor(input_height/model_output_scale),
                                                                   torch.floor(input_width/model_output_scale))
                    local frame_tab = {}
                    for i=1, batch_size do
                        local frame_data = data_tab[indices[i]]
                        local aug_param = aug_param_tab[indices[i]]
                        local frame = image.load(frame_data.filename, 3, 'byte')
                        frame = image.scale(frame, input_width, input_height)

                        -- augment data
                        local aug_frame, aug_annos
                        aug_frame, aug_annos = flipToolPosData(frame, aug_param.flip, frame_data.annotations)
                        aug_frame, aug_annos = rotateToolPos(aug_frame, aug_param.degree, aug_annos)
                        table.insert(frame_tab, aug_frame)
                        -- joint map generation
                        -- note: joint map size is based on the model output: scale down 4 times
                        local jointmap = genJointMapNew(aug_annos, jointNames, j_radius, aug_frame, model_output_scale)
                        frame_batch_jointmap[i] = jointmap:clone()

--                         -- joint map (if generated previously)
--                        frame_batch_jointmap[i] = image.load(frame_data.jointmapname, 1, 'byte')
--                        for j=1, #tooljoint_names do
--                            local joint_annos = frame_annos[tooljoint_names[j]]
--                            if joint_annos ~= nil then
--                                for t_idx=1, #joint_annos do
--                                    local jointmap = image.load(joint_annos[t_idx].jointmapname, 1, 'float')
--                                    frame_batch_jointmap[i][j] = torch.cmax(frame_batch_jointmap[i][j], jointmap)
--                                end
--                            end
--                        end
                    end
                    -- preprocess images
                    local frame_batch = preProcess(frame_tab, input_width, input_height, mean_)
                    collectgarbage()
                    collectgarbage()
                    return frame_batch, frame_batch_jointmap
                end,
                function(frame_batch, frame_batch_jointmap)
                    frame_batch_CPU = frame_batch
                    frame_batch_jointmap_CPU = frame_batch_jointmap
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
        return job_done, frame_batch_CPU, frame_batch_jointmap_CPU
    end

    return loop
end

function DataLoader:terminate()
    self.pool:terminate()
end

return M.DataLoader
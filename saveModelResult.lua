-- this file is used to save the result for visualization or further analysis
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'xlua'
require 'image'
local matio = require 'matio'
require 'data_utils_new'

torch.setdefaulttensortype('torch.FloatTensor')

local SAVE_MAT_FLAG = false
local SAVE_POSE_FLAG = false

local function getDetModelID(modelConf)
    local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
    if modelConf.jointRadius ~= nil then
		s = s .. '_r' .. modelConf.jointRadius
    end
    s = s .. '_whole'
    return s
end

local function getModelID(modelConf)
    local s = modelConf.type
    s = s .. '_v' .. modelConf.v
    s = s .. '_whole'
    return s
end

local function saveMatVisResult(frames, gt_maps, detect_outputs, regress_outputs, joint_names, compo_names, save_dir, indices)
    local batch_size = indices:nElement()
    for idx=1, batch_size do
        local saved_mat = {}
        saved_mat['frame'] = frames[idx]:float()

        local joint_num = #joint_names
        for i=1, joint_num do
            if gt_maps ~= nil then
                saved_mat[string.format('joint_%s_gt', joint_names[i])] = gt_maps[idx][i]:float()
            end
            saved_mat[string.format('conf_%s_result', joint_names[i])] = detect_outputs[idx][i]:float()
            saved_mat[string.format('joint_%s_result', joint_names[i])] = regress_outputs[idx][i]:float()
        end

        local compo_num = #compo_names
        for i=1, compo_num do
            if gt_maps ~= nil then
                saved_mat[string.format('compo_%s_%s_gt', compo_names[i][1], compo_names[i][2])] = gt_maps[idx][joint_num+i]:float()
            end
            saved_mat[string.format('conf_%s_%s_result', compo_names[i][1], compo_names[i][2])] = detect_outputs[idx][joint_num+i]:float()
            saved_mat[string.format('compo_%s_%s_result', compo_names[i][1], compo_names[i][2])] = regress_outputs[idx][joint_num+i]:float()
        end
        matio.save(paths.concat(save_dir, string.format('output_%06d.mat', indices[idx])), saved_mat)
    end
end

local function savePosResult(filename_tab, frames, candidates, poses, joint_names, joint_struture, save_dir, indices, changeFilenameFlag)
    local batch_size = indices:nElement()
    for idx=1, batch_size do
        local saved_mat = {}
        local jointname_saved = {} for i=1, #joint_names do jointname_saved['tool_' .. i] = joint_names[i] end
        local jointstruture_tensor = torch.FloatTensor(joint_struture)
        local pose = poses[idx]
        local candi = candidates[idx]
        local pose_tensor = torch.zeros(#pose, #joint_names+2)
        local candidate_tensor = torch.zeros(#candi, 2+2)
        for i=1, #candi do candidate_tensor[i] = torch.FloatTensor(candi[i]) end
        for tool_idx=1, #pose do
            for i=1, #joint_names+2 do
                pose_tensor[tool_idx][i] = pose[tool_idx][i] or 0
            end
        end

        if changeFilenameFlag then
            local frame_name = filename_tab[idx]
            frame_name = point2newFileLocation(frame_name, '/home/xiaofei/public_datasets', '/Users/xiaofeidu/mData')
            saved_mat['filename'] = frame_name
        else
            saved_mat['filename'] = filename_tab[idx]
        end
--        saved_mat['frame'] = frames[idx]:float()
        saved_mat['pose'] = pose_tensor
        saved_mat['candidates'] = candidate_tensor
        saved_mat['jointName'] = jointname_saved
        saved_mat['jointStructure'] = jointstruture_tensor
        matio.save(paths.concat(save_dir, string.format('pose_%06d.mat', indices[idx])), saved_mat)
    end
end

local function toolPoseEstimate(opt, result_dir, data_type)
    data_type = data_type or 0
    local pose_thres = opt.poseThres or 20
    -- load the detection model
    local detID = getDetModelID(opt.detModelConf)
    local det_model_path = paths.concat(opt.saveDir, 'model.' .. detID .. '.last.t7')
    print('Loading detection model: ' .. det_model_path)
    print(det_model_path)
    local detModel = torch.load(det_model_path)
--    print(detModel)
    detModel:cuda()
    detModel:evaluate()
    local regressID = getModelID(opt.regModelConf)
    local regress_model_path = paths.concat(opt.saveDir, 'model.' .. regressID .. '.last.t7')
    print('Loading regression model: ' .. regress_model_path)
    print(regress_model_path)
    local regressModel = torch.load(regress_model_path)
--    print(regressModel)
    regressModel:cuda()
    regressModel:evaluate()

    local toolJointNum = #opt.toolJointNames
    local toolCompoNum = #opt.toolCompoNames
    local input_width = opt.inputWidth
    local input_height = opt.inputHeight

    -- load data
    local hasGt = true
    local data_file
    if data_type == 1 then
        data_file = paths.concat(opt.dataDir, 'train_endo_toolpos_head.t7')
    elseif data_type == 2 then
        data_file = paths.concat(opt.dataDir, 'val_endo_toolpos_head.t7')
    elseif data_type == 3 then
        data_file = paths.concat(opt.dataDir, 'test_endo_toolpos_head.t7')
    elseif data_type == 4 then
        data_file = paths.concat(opt.dataDir, 'test_old_endo_toolpos_head.t7')
    elseif data_type == 5 then
        data_file = paths.concat(opt.dataDir, 'test_new_endo_toolpos_head.t7')
    elseif data_type == 6 then
        data_file = paths.concat(opt.dataDir, 'test_smoke_toolpos_head.t7')
    else
        data_file = paths.concat(opt.dataDir, 'test_endo_frames.t7')
        hasGt = false
    end
    local data_tab
    if paths.filep(data_file) then
        data_tab = torch.load(data_file)
    else
        error('no such file: ' .. data_file)
    end

    print('Processing data: ' .. data_file)

    local batch_size = opt.batchSize or 1
    local sample_batches = math.floor(#data_tab / batch_size)
    local whole_data_samples = sample_batches * batch_size
    print('There are ' .. whole_data_samples .. ' samples.')
    local data_start_idx = 1
    local data_test_step = 1
    local end_data_index = math.min(whole_data_samples+1, whole_data_samples)

    local order = torch.range(data_start_idx, end_data_index, data_test_step)
    print(string.format('We process from %d to %d, in whole %d samples.', order[1], order[-1], order:size(1)))
    local data_samples = order:size(1)

    local inputsGPU = torch.CudaTensor(batch_size, 3+toolJointNum+toolCompoNum, input_height, input_width)
    local idx = 1
    local recall_tab = {}
    local precision_tab = {}
    local dist_tab = {}
    for i=1, #opt.toolJointNames do
        table.insert(dist_tab, 0.0)
        table.insert(recall_tab, 0.0)
        table.insert(precision_tab, 0.0)
    end
    local wTimer = torch.Timer()
    local timer = torch.Timer()
    local atimer = torch.Timer()
    local ptimer = torch.Timer()
    while idx <= data_samples do
        wTimer:resume()
        ptimer:resume()
        local indices = order:narrow(1, idx, batch_size)
        idx = idx + batch_size
        -- load the frame
        local frame_tab = {}
        local frame_batch_map
        local filename_tab = {}
        local gt_tab = {}
        -- load the ground truth
        if hasGt then
            frame_batch_map = torch.FloatTensor(batch_size, toolJointNum+toolCompoNum,
                                                        torch.floor(input_height/opt.modelOutputScale),
                                                        torch.floor(input_width/opt.modelOutputScale))
            for i=1, batch_size do
                local frame_data = data_tab[indices[i]]
                local frame = image.load(frame_data.filename, 3, 'byte')
                frame = image.scale(frame, input_width, input_height)
                table.insert(frame_tab, frame)
                table.insert(filename_tab, frame_data.filename)
                table.insert(gt_tab, {anno=frame_data.annotations})
                local heatmap = genSepHeatMap(frame_data.annotations, opt.toolJointNames, opt.jointRadius, opt.detJointRadius, frame, opt.modelOutputScale, opt.normalScale)
                frame_batch_map[{i, {1, toolJointNum}}] = heatmap:clone()
                local compmap = genSepPAFMapReg(frame_data.annotations, opt.toolCompoNames, opt.jointRadius, opt.detJointRadius, frame, opt.modelOutputScale, opt.normalScale)
                frame_batch_map[{i, {toolJointNum+1, -1}}] = compmap:clone()
            end
        else
            for i=1, batch_size do
                local frame_data = data_tab[indices[i]]
                local frame = image.load(frame_data.filename, 3, 'byte')
--                frame = image.scale(frame, input_width, input_height)
                table.insert(frame_tab, frame)
                table.insert(filename_tab, frame_data.filename)
            end
        end

        -- preprocess images
        local frameBatchCPU= preProcess(frame_tab, input_width, input_height)

        -- transfer over to GPU
        inputsGPU[{{},{1,3}}]:copy(frameBatchCPU)
        ptimer:stop()
        -- forward
        timer:resume()
        inputsGPU[{{},{4,-1}}] = detModel:forward(inputsGPU[{{},{1,3}}])
        local outputsGPU = regressModel:forward(inputsGPU)
        timer:stop()
        -- tool association
        atimer:resume()
        local candidates, poses = jointAssociation(outputsGPU, opt.toolJointNames, opt.toolTreeStructure, opt.jointRadius, 0.5, 2)
        atimer:stop()
--        print(candidates)
--        print(poses)
        wTimer:stop()
        if hasGt then
            local batch_dist_tab, batch_recall_tab, batch_precision_tab = poseJointPrecision(gt_tab, poses, candidates, opt.toolJointNames, pose_thres, 720, 576)
            for i=1, #opt.toolJointNames do
                if batch_dist_tab[i]~=nil then dist_tab[i] = dist_tab[i] + batch_dist_tab[i] end
                if batch_recall_tab[i]~=nil then recall_tab[i] = recall_tab[i] + batch_recall_tab[i] end
                if batch_precision_tab[i] ~= nil then precision_tab[i] = precision_tab[i] + batch_precision_tab[i] end
            end
        end

        xlua.progress(idx, data_samples)
        if SAVE_MAT_FLAG then saveMatVisResult(frameBatchCPU, frame_batch_map, inputsGPU[{{},{4,-1}}], outputsGPU, opt.toolJointNames, opt.toolCompoNames, result_dir, indices) end
        if SAVE_POSE_FLAG then savePosResult(filename_tab, frameBatchCPU, candidates, poses, opt.toolJointNames, opt.toolTreeStructure, result_dir, indices, true) end
        collectgarbage()
        collectgarbage()
    end
    local preprocess_time = ptimer:time().real
    local cnn_time = timer:time().real
    local asso_time = atimer:time().real
    local whole_time = wTimer:time().real
    print("\nTime to preprocess per sample = " .. preprocess_time/data_samples .. ' sec')
    print("\nTime to cnn per sample = " .. cnn_time/data_samples .. ' sec')
    print("\nTime to associate per sample = " .. asso_time/data_samples .. 'sec')
    print("\nWhole time per sample= " .. whole_time / data_samples .. ' sec')

    if hasGt then
        local dist, rratio, pratio= 0.0, 0.0, 0.0 -- average joint prec
        for i=1, #opt.toolJointNames do
            dist_tab[i] = dist_tab[i] / data_samples
            recall_tab[i] = recall_tab[i] / data_samples
            precision_tab[i] = precision_tab[i] / data_samples
            print(string.format('%s average  dist     =  %.2f',opt.toolJointNames[i], dist_tab[i]))
            print(string.format('%s average recall    = %.2f %%',opt.toolJointNames[i], 100 * recall_tab[i]))
            print(string.format('%s average precision = %.2f %%',opt.toolJointNames[i], 100 * precision_tab[i]))
            dist = dist + dist_tab[i]
            rratio = rratio + recall_tab[i]
            pratio = pratio + precision_tab[i]

        end
        dist = dist / #opt.toolJointNames
        rratio = rratio / #opt.toolJointNames
        pratio = pratio / #opt.toolJointNames
        print(string.format('All average dist for all joints      =  %.2f', dist))
        print(string.format('All average recall for all joints    = %.2f %%', 100 * rratio))
        print(string.format('All average precision for all joints = %.2f %%', 100 * pratio))
    end
end

local opt = {
    dataDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label',
	saveDir = '/home/xiaofei/workspace/toolPose/models',
--	detModelConf = {type='toolPartDetFull', v=1, jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256},
--	regModelConf = {type='toolPoseRegressFull', v=2, jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10},
--	detModelConf = {type='toolPartDetFull', v='256*320_ftblr_head', jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256},
--    regModelConf = {type='toolPoseRegressFull', v='256*320_ftblr_head', jointRadius=5, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=1, hflip=1},

    -- MICCAI
    detModelConf = {type='toolPartDetFull', v='256*320_ftblr_head', jointRadius=15, modelOutputScale=1, inputWidth=320, inputHeight=256},
    regModelConf = {type='toolPoseRegressFull', v='256*320_ftblr_head_noConcat', jointRadius=20, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=1, hflip=1},

--    detModelConf = {type='toolPartDetFull', v='256*320_ftblr_random_head', jointRadius=15, modelOutputScale=1, inputWidth=320, inputHeight=256},
--    regModelConf = {type='toolPoseRegressFull', v='256*320_ftblr_random_head_noConcat', jointRadius=20, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10, vflip=0, hflip=0},



    gpus = {1},
	batchSize = 1,
    toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint', 'EndPoint' }, -- joint number = 5
	toolCompoNames = {{'LeftClasperPoint', 'HeadPoint'},
					  {'RightClasperPoint', 'HeadPoint'},
					  {'HeadPoint', 'ShaftPoint'},
                      {'ShaftPoint', 'EndPoint'}
					 },
    toolTreeStructure = {{1, 3},{2, 3},{3, 4},{4, 5} },
    poseThres = 20,
}
opt.inputWidth = opt.regModelConf.inputWidth or 320 -- 480  -- 720
opt.inputHeight = opt.regModelConf.inputHeight or 256 -- 384 -- 576
opt.modelOutputScale = opt.regModelConf.modelOutputScale or 4
opt.detJointRadius = opt.detModelConf.jointRadius or 10
opt.jointRadius = opt.regModelConf.jointRadius or 20
opt.normalScale = opt.regModelConf.normalScale or 1

if not paths.dirp(opt.dataDir) then
	error("Can't find directory : " .. opt.dataDir)
end

local resultDir = '/home/xiaofei/workspace/toolPose/visual_results/test'
if not paths.dirp(resultDir) then
	os.execute('mkdir -p ' .. resultDir)
end

toolPoseEstimate(opt, resultDir, 6)

-- copy /home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing/test_endo_frames.t7 to
-- /home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label/test_endo_frames.t7
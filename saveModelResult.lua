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


local function getModelID(modelConf)
    local s = modelConf.type
    if modelConf.iterCnt ~= nil then
        s = s .. '_i' .. modelConf.iterCnt
    end
    s = s .. '_v' .. modelConf.v
    return s
end

local function saveMatResult(frames, gt_maps, detect_outputs, regress_outputs, joint_names, compo_names, save_dir, indices)
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

local function toolPoseEstimate(opt, result_dir, data_type)
    data_type = data_type or 3
    -- load the detection model
    print('Loading detection model ...')
    local detID = getModelID(opt.detModelConf)
    local det_model_path = paths.concat(opt.saveDir, 'model.' .. detID .. '.best.t7');
    local detModel = torch.load(det_model_path)
    print(detModel)
    detModel:cuda()
    detModel:evaluate()
    print('Loading regression model...')
    local regressID = getModelID(opt.regModelConf)
    local regress_model_path = paths.concat(opt.saveDir, 'model.' .. regressID .. '.best.t7');
    local regressModel = torch.load(regress_model_path)
    print(regressModel)
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
        data_file = paths.concat(opt.dataDir, 'train_endo_toolpos.t7')
    elseif data_type == 2 then
        data_file = paths.concat(opt.dataDir, 'val_endo_toolpos.t7')
    elseif data_type == 3 then
        data_file = paths.concat(opt.dataDir, 'test_endo_frames.t7')
        hasGt = false
    end
    local data_tab
    if paths.filep(data_file) then
        data_tab = torch.load(data_file)
    else
        error('no such file: ' .. data_file)
    end

    local batch_size = opt.batchSize or 1
    local sample_batches = math.floor(#data_tab / batch_size)
    local data_samples = sample_batches * batch_size
    local data_start_idx = 1500
    local data_test_step = 10
    data_samples = 150

    local order = torch.range(1, data_samples*data_test_step, data_test_step) + data_start_idx

    local inputsGPU = torch.CudaTensor(batch_size, 3+toolJointNum+toolCompoNum, input_height, input_width)
    local idx = 1
    local processed_time = 0
    local wTimer = torch.Timer()
    local timer = torch.Timer()
    while idx <= data_samples do
        timer:reset()
        local indices = order:narrow(1, idx, batch_size)
        idx = idx + batch_size
        -- load the frame
        local frame_tab = {}
        local frame_batch_map
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
                local heatmap = genSepHeatMap(frame_data.annotations, opt.toolJointNames, opt.jointRadius, opt.detJointRadius, frame, opt.modelOutputScale, opt.normalScale)
                frame_batch_map[{i, {1, toolJointNum}}] = heatmap:clone()
                local compmap = genSepPAFMapReg(frame_data.annotations, opt.toolCompoNames, opt.jointRadius, opt.detJointRadius, frame, opt.modelOutputScale, opt.normalScale)
                frame_batch_map[{i, {toolJointNum+1, -1}}] = compmap:clone()
            end
        else
            for i=1, batch_size do
                local frame_data = data_tab[indices[i]]
                local frame = image.load(frame_data.filename, 3, 'byte')
                frame = image.scale(frame, input_width, input_height)
                table.insert(frame_tab, frame)
            end
        end

        -- preprocess images
        local frameBatchCPU= preProcess(frame_tab, input_width, input_height)

        -- transfer over to GPU
        inputsGPU[{{},{1,3}}]:copy(frameBatchCPU)

        -- forward
        inputsGPU[{{},{4,-1}}] = detModel:forward(inputsGPU[{{},{1,3}}])
        local outputsGPU = regressModel:forward(inputsGPU)

        processed_time = processed_time + timer:time().real
        xlua.progress(idx, data_samples)
        saveMatResult(frameBatchCPU, frame_batch_map, inputsGPU[{{},{4,-1}}], outputsGPU, opt.toolJointNames, opt.toolCompoNames, result_dir, indices)
        collectgarbage()
        collectgarbage()
    end
    print("\nTime to process per sample = " .. processed_time/data_samples .. ' sec')
    print("\nWhole time = " .. wTimer:time().real .. ' sec')
end

local opt = {
    dataDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label',
	saveDir = '/home/xiaofei/workspace/toolPose/models',
--    detModelConf = {type='toolDualPoseSep', v=1, jointRadius=20, modelOutputScale=4},
--	regModelConf = {type='toolPoseRegress', v=1, jointRadius = 20, modelOutputScale=4},
	detModelConf = {type='toolPartDetFull', v=1, jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256},
	regModelConf = {type='toolPoseRegressFull', v=2, jointRadius=10, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10},
--	regModelConf = {type='toolPoseRegressFull', v=4, jointRadius=20, modelOutputScale=1, inputWidth=320, inputHeight=256, normalScale=10},
	gpus = {1},
	batchSize = 1,
    toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint', 'EndPoint' }, -- joint number = 5
	toolCompoNames = {{'LeftClasperPoint', 'HeadPoint'},
					  {'RightClasperPoint', 'HeadPoint'},
					  {'HeadPoint', 'ShaftPoint'},
                      {'ShaftPoint', 'EndPoint'}
					 },
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

toolPoseEstimate(opt, resultDir)

-- copy /home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing/test_endo_frames.t7 to
-- /home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label/test_endo_frames.t7
require 'image'
require 'colormap'
require 'data_utils_new'
local matio = require 'matio'

local function visualResult(frames, gt_maps, outputs_map, joint_names, compo_names, saveDir)
    local batch_size = gt_maps:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), frames[rand_idx]:byte())
    local joint_num = #joint_names
    for i=1, joint_num do
        image.save(paths.concat(saveDir, string.format('frame_%s_reg_gt.png', joint_names[i])), (255*torch.abs(gt_maps[rand_idx][i])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_%s_result.png', joint_names[i])), (255*torch.abs(outputs_map[rand_idx][i])):byte())
    end

    local compo_num = #compo_names
    for i=1, compo_num do
--        print(gt_maps[rand_idx][joint_num+2*i-1]:max())
--        print(torch.abs(gt_maps[rand_idx][joint_num+2*i]):max())

        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_reg_gt.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+i]):byte())

--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gtx.png', compo_names[i][1], compo_names[i][2])), (255*torch.abs(gt_maps[rand_idx][joint_num+2*i-1])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gty.png', compo_names[i][1], compo_names[i][2])), (255*torch.abs(gt_maps[rand_idx][joint_num+2*i])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resultx.png', compo_names[i][1],compo_names[i][2])), (255*torch.abs(outputs_map[rand_idx][joint_num+2*i-1])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resulty.png', compo_names[i][1],compo_names[i][2])), (255*torch.abs(outputs_map[rand_idx][joint_num+2*i])):byte())
    end
end

local function saveMatResult(frames, gt_maps, detect_outputs_map, regress_outputs_map, joint_names, compo_names, saveDir)
    detect_outputs_map:clamp(0,1)
--    regress_outputs_map:clamp(0,1)
    local batch_size = frames:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    local saved_mat = {}
    saved_mat['frame'] = frames[rand_idx]:float()
--    matio.save(paths.concat(saveDir, 'frame_raw.mat'), {frame=frames[rand_idx]:float()});

    local joint_num = #joint_names
    for i=1, joint_num do
--        print(gt_maps[rand_idx][i]:min(), gt_maps[rand_idx][i]:max())
--        print(outputs_map[rand_idx][i]:min(), outputs_map[rand_idx][i]:max())
        if gt_maps ~= nil then
            saved_mat[string.format('joint_%s_gt', joint_names[i])] = gt_maps[rand_idx][i]:float()
--            matio.save(paths.concat(saveDir, string.format('frame_%s_gt.mat', joint_names[i])), gt_maps[rand_idx][i]:float());
        end
        saved_mat[string.format('conf_%s_result', joint_names[i])] = detect_outputs_map[rand_idx][i]:float()
        saved_mat[string.format('joint_%s_result', joint_names[i])] = regress_outputs_map[rand_idx][i]:float()
--        matio.save(paths.concat(saveDir, string.format('frame_%s_result.mat', joint_names[i])), outputs_map[rand_idx][i]:float())
    end

    local compo_num = #compo_names
    for i=1, compo_num do
--        print(gt_maps[rand_idx][joint_num+i]:min(), gt_maps[rand_idx][joint_num+i]:max())
--        print(outputs_map[rand_idx][joint_num+i]:min(), outputs_map[rand_idx][joint_num+i]:max())

        if gt_maps ~= nil then
            saved_mat[string.format('compo_%s_%s_gt', compo_names[i][1], compo_names[i][2])] = gt_maps[rand_idx][joint_num+i]:float()
--            matio.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gt.mat', compo_names[i][1], compo_names[i][2])), gt_maps[rand_idx][joint_num+i]:float())
        end
        saved_mat[string.format('conf_%s_%s_result', compo_names[i][1], compo_names[i][2])] = detect_outputs_map[rand_idx][joint_num+i]:float()
        saved_mat[string.format('compo_%s_%s_result', compo_names[i][1], compo_names[i][2])] = regress_outputs_map[rand_idx][joint_num+i]:float()
--        matio.save(paths.concat(saveDir, string.format('frame_{%s_%s}_result.mat', compo_names[i][1], compo_names[i][2])), outputs_map[rand_idx][joint_num+i]:float())
    end
    matio.save(paths.concat(saveDir, 'output.mat'), saved_mat)
end

--local baseDir = '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/tool_label'
local baseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label'
local train_anno_tab = torch.load(paths.concat(baseDir, 'train_endo_toolpos_head.t7'))
print(#train_anno_tab)

local input_height = 256 --256   -- 512    -- 576    -- 384
local input_width = 320 --320     -- 640     --720     -- 480
local frame_data = train_anno_tab[272]
local frame = image.load(frame_data.filename, 3, 'byte')
print(frame_data.filename)
frame = image.scale(frame, input_width, input_height)

local joint_names = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint', 'EndPoint' } -- joint number = 5
local compo_names = {{'LeftClasperPoint', 'HeadPoint'},
					  {'RightClasperPoint', 'HeadPoint'},
					  {'HeadPoint', 'ShaftPoint'},
                      {'ShaftPoint', 'EndPoint'}}
local scale = 1
local det_mapsCPU = torch.FloatTensor(1, 5+4, torch.floor(input_height/scale), torch.floor(input_width/scale))
local reg_mapsCPU = torch.FloatTensor(1, 5+4, torch.floor(input_height/scale), torch.floor(input_width/scale))

local aug_frame, aug_annos = flipToolPosData(frame, 0, 0, frame_data.annotations)
aug_frame, aug_annos = rotateToolPos(aug_frame, 0, aug_annos)

local joint_radius = 15
local reg_radius = 20

local joint_map = genSepJointMap(aug_annos, joint_names, joint_radius, aug_frame, scale)
local heat_map = genSepHeatMap(aug_annos, joint_names, reg_radius, joint_radius, aug_frame, scale, 10)
local comp_map = genSepPAFMapDet(aug_annos, compo_names, joint_radius, aug_frame, scale)
local comp_heatmap = genSepPAFMapReg(aug_annos, compo_names, reg_radius, joint_radius, aug_frame, scale, 10)

det_mapsCPU[{1, {1, 5}}] = joint_map:clone()
det_mapsCPU[{1, {6,-1}}] = comp_map:clone()

reg_mapsCPU[{1, {1, 5}}] = heat_map:clone()
reg_mapsCPU[{1, {6,-1}}] = comp_heatmap:clone()


-- visual
local framesCPU = aug_frame:reshape(1, 3, input_height, input_width)
--local outputsGPU = det_mapsCPU
--local save_dir = '/Users/xiaofeidu/workspace/toolpose_results/foo'
local save_dir = '/home/xiaofei/workspace/toolPose/foo'
--visualResult(framesCPU, mapsCPU, outputsGPU, joint_names, compo_names, save_dir)

saveMatResult(framesCPU, nil, det_mapsCPU, reg_mapsCPU, joint_names, compo_names, save_dir)

--local cp_colormap = colormap:convert(heat_map:float())
--print(cp_colormap:max())
--cp_colormap = 255 * cp_colormap
--local fused_frame = aug_frame:float() * 0.7 + cp_colormap:float() * 0.3
--image.save(paths.concat(baseDir, string.format('fused_%d_%d.png', input_width, input_height)), fused_frame:byte())
--
--image.save(paths.concat(baseDir, string.format('heatmap_%d_%d.png', input_width, input_height)), cp_colormap:byte())

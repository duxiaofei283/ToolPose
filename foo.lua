require 'image'
require 'colormap'
require 'data_utils_new'

local function visualResult(frames, gt_maps, outputs_map, joint_names, compo_names, saveDir)
    local batch_size = gt_maps:size(1)
    local rand_idx = torch.ceil(batch_size * math.random())
    image.save(paths.concat(saveDir, 'frame_raw.png'), frames[rand_idx]:byte())
    local joint_num = #joint_names
    for i=1, joint_num do
        image.save(paths.concat(saveDir, string.format('frame_%s_gt.png', joint_names[i])), (255*torch.abs(gt_maps[rand_idx][i])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_%s_result.png', joint_names[i])), (255*torch.abs(outputs_map[rand_idx][i])):byte())
    end

    local compo_num = #compo_names
    for i=1, compo_num do
--        print(gt_maps[rand_idx][joint_num+2*i-1]:max())
--        print(torch.abs(gt_maps[rand_idx][joint_num+2*i]):max())

        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gt.png', compo_names[i][1], compo_names[i][2])), (255*gt_maps[rand_idx][joint_num+i]):byte())

--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gtx.png', compo_names[i][1], compo_names[i][2])), (255*torch.abs(gt_maps[rand_idx][joint_num+2*i-1])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_gty.png', compo_names[i][1], compo_names[i][2])), (255*torch.abs(gt_maps[rand_idx][joint_num+2*i])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resultx.png', compo_names[i][1],compo_names[i][2])), (255*torch.abs(outputs_map[rand_idx][joint_num+2*i-1])):byte())
--        image.save(paths.concat(saveDir, string.format('frame_{%s_%s}_resulty.png', compo_names[i][1],compo_names[i][2])), (255*torch.abs(outputs_map[rand_idx][joint_num+2*i])):byte())
    end
end

local baseDir = '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/tool_label'
local train_anno_tab = torch.load(paths.concat(baseDir, 'train_endo_toolpos.t7'))

local input_height = 384   -- 512    -- 576    -- 384
local input_width = 480     -- 640     --720     -- 480
local frame_data = train_anno_tab[20]
local frame = image.load(frame_data.filename, 3, 'byte')
frame = image.scale(frame, input_width, input_height)

local joint_names = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint', 'EndPoint' } -- joint number = 5
local compo_names = {{'LeftClasperPoint', 'HeadPoint'},
					  {'RightClasperPoint', 'HeadPoint'},
					  {'HeadPoint', 'ShaftPoint'},
                      {'ShaftPoint', 'EndPoint'}}
local scale = 4
local mapsCPU = torch.FloatTensor(1, 5+4, torch.floor(input_height/scale), torch.floor(input_width/scale))

local aug_frame, aug_annos = flipToolPosData(frame, 0, frame_data.annotations)
aug_frame, aug_annos = rotateToolPos(aug_frame, 0, aug_annos)

local heat_map = genSepHeatMap(aug_annos, joint_names, 20, aug_frame, scale)
local joint_map = genSepJointMap(aug_annos, joint_names, 20, aug_frame, scale)
local comp_map = genSepPAFMap(aug_annos, compo_names, 20, aug_frame, scale)
local comp_heatmap = genSepPAFMapReg(aug_annos, compo_names, 20, aug_frame, scale)

mapsCPU[{1, {1, 5}}] = heat_map:clone()
mapsCPU[{1, {6,-1}}] = comp_heatmap:clone()

-- visual
local framesCPU = frame:reshape(1, 3, input_height, input_width)
local outputsGPU = mapsCPU
visualResult(framesCPU, mapsCPU, outputsGPU, joint_names, compo_names,'/Users/xiaofeidu/workspace/toolpose_results/foo')



--local cp_colormap = colormap:convert(heat_map:float())
--print(cp_colormap:max())
--cp_colormap = 255 * cp_colormap
--local fused_frame = aug_frame:float() * 0.7 + cp_colormap:float() * 0.3
--image.save(paths.concat(baseDir, string.format('fused_%d_%d.png', input_width, input_height)), fused_frame:byte())
--
--image.save(paths.concat(baseDir, string.format('heatmap_%d_%d.png', input_width, input_height)), cp_colormap:byte())

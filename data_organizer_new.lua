
-- normalize the pos to [0,1]
require 'json'
require 'image'
require 'colormap'
require 'data_utils_new'
torch.setdefaulttensortype('torch.FloatTensor')

-- read json file
-- keyword
-- filename, class=image, annotations
-- for annotations: keyword
--  class = LeftClasperPoint, RightClasperPoint HeadPoint, ShaftPoint, TrackedPoint, EndPoint.
--  id = tool1, tool2
--  x y
local function readtoolLabelFile(label_file_tab)
    local file_num = #label_file_tab
    for seq_idx=1, file_num do
        local jsonFilePath = label_file_tab[seq_idx]
        local json_tab = json.load(jsonFilePath)

        local frame_num = #json_tab
        local anno_tab = {}
        local anno_frame_num = 0
        -- frame
        for i=1, frame_num do
            local frame_name = json_tab[i].filename
            print('old frame ' .. frame_name)

            -- point to new file location
            frame_name = point2newFileLocation(frame_name, '/Users/xiaofeidu/mData', '/home/xiaofei/public_datasets')
            print('new frame ' .. frame_name)

            local annotations = json_tab[i].annotations
            if #annotations ~= 0 then
                anno_frame_num = anno_frame_num + 1
                anno_tab[anno_frame_num] = {}
                anno_tab[anno_frame_num].filename = frame_name
                local tool_ids = {}
                -- reformat annotations: using joint class as key
                local frame_anno = {}
                for j=1, #annotations do
                    local joint_anno = annotations[j]

                    if frame_anno[joint_anno.class] == nil then
                        frame_anno[joint_anno.class] = {}
                    end

                    table.insert(frame_anno[joint_anno.class], { id = joint_anno.id,
                                                                  x = joint_anno.x,
                                                                  y = joint_anno.y
                                                               }
                    )

                    tool_ids[joint_anno.id] = true
                end
                anno_tab[anno_frame_num].annotations = frame_anno
                anno_tab[anno_frame_num].jointNum = #annotations

                local tool_num = 0
                for __, __ in pairs(tool_ids) do
                    tool_num = tool_num + 1
                end

                anno_tab[anno_frame_num].toolNum = tool_num

            end
        end

        -- normalize the location
        for i=1, #anno_tab do
            local frame_name = anno_tab[i].filename
            local frame = image.load(frame_name, 3, 'byte')
            local frame_width = frame:size(3)
            local frame_height = frame:size(2)
            local norm_frame_anno = normalizeToolPos01(frame_width, frame_height, anno_tab[i].annotations)
            anno_tab[i].annotations = norm_frame_anno
        end
        return anno_tab
    end
end
local function sepTrainingData(anno_tab)
    local anno_frame_num = #anno_tab
    assert(anno_frame_num >= 1)

    local train_anno_tab = {}
    local val_anno_tab = {}

    local train_anno_frame_num = math.max(math.floor(0.8 * anno_frame_num), 1)

    for i=1, train_anno_frame_num do
        table.insert(train_anno_tab, anno_tab[i])
    end
    for i=train_anno_frame_num+1, anno_frame_num do
        table.insert(val_anno_tab, anno_tab[i])
    end
    return train_anno_tab, val_anno_tab
end
-- seq_info_tab = {{seqDir=, frameFormat=, startFrame=, endFrame=}}
local function genTestData(seq_info_tab)
    local seq_num = #seq_info_tab
    local anno_tab = {}
    for seq_idx=1, seq_num do
        local seq_info = seq_info_tab[seq_idx]
        local frame_num = seq_info.endFrame - seq_info.startFrame + 1
        local frame_anno = {}
        for frame_idx=1, frame_num do
            frame_anno.filename = paths.concat(seq_info.seqDir, string.format(seq_info.frameFormat, frame_idx))
            frame_anno.annotations = nil
            frame_anno.jointNum = nil
            frame_anno.toolNum = nil
            table.insert(anno_tab, frame_anno)
        end
    end

    return anno_tab
end


--local trainBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label'
--local scale = 4
--local joint_names = {'LeftClasperPoint', 'RightClasperPoint',
--                          'HeadPoint', 'ShaftPoint',
--                          'TrackedPoint', 'EndPoint' } -- joint number = 6
--local json_files = {}
--for seq_idx=1, 4 do
--    local json_file_path = paths.concat(trainBaseDir, 'endo' .. seq_idx .. '_labels.json')
--    table.insert(json_files, json_file_path)
--end
--local anno_tab = readtoolLabelFile(json_files)
--local train_anno_tab, val_anno_tab = sepTrainingData(anno_tab)
--torch.save(paths.concat(trainBaseDir, 'train_endo_toolpos.t7'), train_anno_tab)
--torch.save(paths.concat(trainBaseDir, 'val_endo_toolpos.t7'), val_anno_tab)
--print('===========================================================================')

local seq_info_tab = {}
local testBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing'
local frame_format = 'img_%06d_raw.png'
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset1', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=370})
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset2', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375})
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset3', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375})
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset4', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375})
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset5', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1500})
table.insert(seq_info_tab, {seqDir=paths.concat(testBaseDir, 'Dataset6', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1500})
local test_anno_tab = genTestData(seq_info_tab)
torch.save(paths.concat(testBaseDir, 'test_endo_frames.t7'), test_anno_tab)


---- ---------------------------------------------------------------------------------------
--
---- data augmentation
---- raw image format: img_0000xx_raw.png
---- gen joint map and heat map
--local radius = 10
--local sigma = 20
--local maxDegree = 5
--local train_aug_anno_tab = {}
--local val_aug_anno_tab = {}
--
--local data_tabs = {train_anno_tab, val_anno_tab}
--local aug_data_tabs = {train_aug_anno_tab, val_aug_anno_tab}
--
--for xx = 1, #aug_data_tabs do
--    local aug_data_tab = aug_data_tabs[xx]
--    local data_tab = data_tabs[xx]
--
--    for i=1, #data_tab do
--        local frame_anno_tab = data_tab[i]
--        local frame_name = frame_anno_tab.filename
--        print('Augment frame ' .. frame_name)
--        local frame = image.load(frame_name, 3, 'byte')
--        local frame_extname = paths.extname(frame_name)
--        local frame_dir = paths.dirname(frame_name)
--        local aug_dir = paths.concat(paths.dirname(frame_dir), 'aug')
--        if not paths.dirp(aug_dir) then
--            paths.mkdir(aug_dir)
--        end
--
--        local frame_basename = paths.basename(frame_name, frame_extname)
--        local frame_idx = tonumber(string.match(frame_basename, '%d+'))
--
--        for degree = -1*maxDegree, maxDegree do  -- rotation
--            for flip = 0, 0 do  -- no flip and flip
--                -- augmented frame
--                local aug_frame, aug_annos
--                local aug_frame_name_format = 'img_%06d_f%d_d%d_raw.%s'
--                local aug_frame_name = paths.concat(aug_dir, string.format(aug_frame_name_format, frame_idx, flip, degree, frame_extname))
--
--                -- FLIP IS WRONG!!!: not realistic ... need to improve, more data?
--                aug_frame, aug_annos = flipToolPosData(frame, flip, frame_anno_tab.annotations)
--                aug_frame, aug_annos = rotateToolPos(aug_frame, degree, aug_annos)
--
--                image.save(aug_frame_name, aug_frame)
--
--                -- augmented jointmap and heatmap
--                local joint_map, heat_map, jointmap_name, heatmap_name
----                local jointmap_name_format = 'img_%06d_f%d_d%d_%s_jointmap.%s'  -- tool specific
--                local jointmap_name_format = 'img_%06d_f%d_d%d_jointmap.%s'
--                local heatmap_name_format =  'img_%06d_f%d_d%d_%s_%s_heatmap.%s'
--
--                jointmap_name = paths.concat(aug_dir, string.format(jointmap_name_format, frame_idx, flip, degree, frame_extname))
--                joint_map = genJointMapNew(aug_annos, joint_names, radius, aug_frame, scale)
----                local j_colormap = colormap:convert(joint_map:float())
--                image.save(jointmap_name, joint_map)
--
--
--                for joint_class, __ in pairs(aug_annos) do
--                    local joint_anno = aug_annos[joint_class]
--                    for tool_idx = 1, #joint_anno do
--                        local joint_id = joint_anno[tool_idx].id
--                        heatmap_name = paths.concat(aug_dir, string.format(heatmap_name_format, frame_idx, flip, degree, joint_id, joint_class, frame_extname))
----                      print(string.format('ori joint_pos: [%f, %f]', frame_anno_tab.annotations[joint_class].x, frame_anno_tab.annotations[joint_class].y))
----                      print(string.format('aug joint_pos: [%f, %f]', joint_anno.x, joint_anno.y))
----                      heat_map = genHeatMapFast(joint_anno.x, joint_anno.y, sigma, aug_frame, scale)
----                      local colormap_ = colormap:convert(heat_map)
----                      image.save(heatmap_name, colormap_)
--                        joint_anno[tool_idx].heatmapname = heatmap_name
--                    end
--                end
--
--                local aug_frame_anno_tab = {}
--                aug_frame_anno_tab.jointNum = frame_anno_tab.jointNum
--                aug_frame_anno_tab.toolNum = frame_anno_tab.toolNum
--
--                aug_frame_anno_tab.filename = aug_frame_name
--                aug_frame_anno_tab.jointmapname = jointmap_name
--                aug_frame_anno_tab.annotations = aug_annos
--
--                table.insert(aug_data_tab, aug_frame_anno_tab)
--            end
--        end
--    end
--end
--
--
--
--torch.save(paths.concat(baseDir, 'train_endo_aug_toolpos.t7'), train_aug_anno_tab)
--torch.save(paths.concat(baseDir, 'val_endo_aug_toolpos.t7'), val_aug_anno_tab)





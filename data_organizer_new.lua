
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
    local multi_seq_anno_tab = {}

    for seq_idx=1, file_num do
        local jsonFilePath = label_file_tab[seq_idx]
        local json_tab = json.load(jsonFilePath)

        local frame_num = #json_tab
        local anno_tab = {}
        local anno_frame_num = 0
        -- frame
        for i=1, frame_num do
            local frame_name = json_tab[i].filename
--            print('old frame ' .. frame_name)

            -- point to new file location
            frame_name = point2newFileLocation(frame_name, '/Users/xiaofeidu/mData', '/home/xiaofei/public_datasets')
            frame_name = changeFrameFormat(frame_name, 'img_%06d_raw.png')
--            print('new frame ' .. frame_name)

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
        table.insert(multi_seq_anno_tab, anno_tab)
    end

    return multi_seq_anno_tab
end

local function readOldToolLabelFile(label_file_tab)
    local file_num = #label_file_tab
    local multi_seq_anno_tab = {}

    for seq_idx=1, file_num do
        local jsonFilePath = label_file_tab[seq_idx]
        local json_tab = json.load(jsonFilePath)

        local frame_num = #json_tab
        local anno_tab = {}
        local anno_frame_num = 0
        -- frame
        for i=1, frame_num do
            local frame_name = json_tab[i].filename
--            print('old frame ' .. frame_name)

            -- point to new file location
            frame_name = point2newFileLocation(frame_name, '/Users/xiaofeidu/mData', '/home/xiaofei/public_datasets')
            frame_name = changeFrameFormat(frame_name, 'img_%06d_raw.png')
--            print('new frame ' .. frame_name)

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

                    if joint_anno.id == 'tool1' then
                        table.insert(frame_anno[joint_anno.class], { id = joint_anno.id,
                                                                  x = joint_anno.x,
                                                                  y = joint_anno.y
                                                               }
                        )
                        tool_ids[joint_anno.id] = true
                    end
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
        table.insert(multi_seq_anno_tab, anno_tab)
    end

    return multi_seq_anno_tab
end

local function readNewToolLabelFile(label_file_tab)
    local file_num = #label_file_tab
    local multi_seq_anno_tab = {}

    for seq_idx=1, file_num do
        local jsonFilePath = label_file_tab[seq_idx]
        local json_tab = json.load(jsonFilePath)

        local frame_num = #json_tab
        local anno_tab = {}
        local anno_frame_num = 0
        -- frame
        for i=1, frame_num do
            local frame_name = json_tab[i].filename
--            print('old frame ' .. frame_name)

            -- point to new file location
            frame_name = point2newFileLocation(frame_name, '/Users/xiaofeidu/mData', '/home/xiaofei/public_datasets')
            frame_name = changeFrameFormat(frame_name, 'img_%06d_raw.png')
--            print('new frame ' .. frame_name)

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

                    if joint_anno.id == 'tool2' then
                        table.insert(frame_anno[joint_anno.class], { id = joint_anno.id,
                                                                  x = joint_anno.x,
                                                                  y = joint_anno.y
                                                                    }
                        )

                        tool_ids[joint_anno.id] = true
                    end
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
        table.insert(multi_seq_anno_tab, anno_tab)
    end

    return multi_seq_anno_tab
end

-- seperate the data into train and validation set for single sequence
local function sepTrainingData(anno_tab, train_percentage)
    train_percentage = train_percentage or 0.8
    local anno_frame_num = #anno_tab
    assert(anno_frame_num >= 1)

    local train_anno_tab = {}
    local val_anno_tab = {}

    local train_anno_frame_num = math.max(math.floor(train_percentage * anno_frame_num), 1)

    for i=1, train_anno_frame_num do
        table.insert(train_anno_tab, anno_tab[i])
    end
    for i=train_anno_frame_num+1, anno_frame_num do
        table.insert(val_anno_tab, anno_tab[i])
    end
    return train_anno_tab, val_anno_tab
end

-- seperate the data into train and validation set for multiple sequence (internal sequence 80% : 20%)
local function internalSepTrainingData(multi_seq_anno_tab, train_percentage)
    train_percentage = train_percentage or 0.8
    local seq_num = #multi_seq_anno_tab
    local train_anno_tab = {}
    local val_anno_tab = {}
    for seq_idx=1, seq_num do
        local anno_tab = multi_seq_anno_tab[seq_idx]
        local anno_frame_num = #anno_tab
        assert(anno_frame_num >= 1)

        local train_anno_frame_num = math.max(math.floor(train_percentage * anno_frame_num), 1)

        for i=1, train_anno_frame_num do
            table.insert(train_anno_tab, anno_tab[i])
        end
        for i=train_anno_frame_num+1, anno_frame_num do
            table.insert(val_anno_tab, anno_tab[i])
        end
        print(train_anno_frame_num, anno_frame_num - train_anno_frame_num)
    end
    return train_anno_tab, val_anno_tab
end

local function internalRandomSepTrainingData(multi_seq_anno_tab, train_percentage)
    train_percentage = train_percentage or 0.8
    local seq_num = #multi_seq_anno_tab
    local train_anno_tab = {}
    local val_anno_tab = {}
    for seq_idx=1, seq_num do
        local anno_tab = multi_seq_anno_tab[seq_idx]
        local anno_frame_num = #anno_tab
        assert(anno_frame_num >= 1)
        local perm = torch.randperm(anno_frame_num)

        local train_anno_frame_num = math.max(math.floor(train_percentage * anno_frame_num), 1)

        for i=1, train_anno_frame_num do
            table.insert(train_anno_tab, anno_tab[perm[i]])
        end
        for i=train_anno_frame_num+1, anno_frame_num do
            table.insert(val_anno_tab, anno_tab[perm[i]])
        end
        print(train_anno_frame_num, anno_frame_num - train_anno_frame_num)
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
        for frame_idx=1, frame_num do
            local frame_anno = {}
            frame_anno.filename = paths.concat(seq_info.seqDir, string.format(seq_info.frameFormat, frame_idx))
            frame_anno.annotations = nil
            frame_anno.jointNum = nil
            frame_anno.toolNum = nil
            table.insert(anno_tab, frame_anno)
        end
    end

    return anno_tab
end

local joint_names = {'LeftClasperPoint', 'RightClasperPoint',
                          'HeadPoint', 'ShaftPoint',
                          'TrackedPoint', 'EndPoint' } -- joint number = 6

-- train dataset
--local trainBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label'
--local json_files = {}
--for seq_idx=1, 4 do
----    local json_file_path = paths.concat(trainBaseDir, 'endo' .. seq_idx .. '_labels.json')  -- original label
--    local json_file_path = paths.concat(trainBaseDir, 'train' .. seq_idx .. '_labels.json')  -- improved label (head)
--    table.insert(json_files, json_file_path)
--end
--local anno_tab = readtoolLabelFile(json_files)
--local train_anno_tab, val_anno_tab = internalSepTrainingData(anno_tab)
--print(#train_anno_tab)
--print(#val_anno_tab)
--torch.save(paths.concat(trainBaseDir, 'train_endo_toolpos_head.t7'), train_anno_tab)
--torch.save(paths.concat(trainBaseDir, 'val_endo_toolpos_head.t7'), val_anno_tab)
--print('===========================================================================')

-- ---------------------------------------------------------------------------------------------------
 -- in vivo
local invivoBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/tool_label'
local invivo_json_files = {}
for seq_idx = 1, 3 do
    local json_file_path = paths.concat(invivoBaseDir, 'invivo' .. seq_idx .. '_labels.json')
    table.insert(invivo_json_files, json_file_path)
end

--local anno_tab = readtoolLabelFile(invivo_json_files)
--local train_anno_tab, val_anno_tab = internalRandomSepTrainingData(anno_tab, 0.8)
--print(#train_anno_tab, #val_anno_tab)
--torch.save(paths.concat(invivoBaseDir, 'train_invivo2_toolpos.t7'), train_anno_tab)
--torch.save(paths.concat(invivoBaseDir, 'val_invivo2_toolpos.t7'), val_anno_tab)

local exvivoBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing/tool_label'
local exvivo_json_files = {}
for seq_dix = 1, 6 do
    local json_file_path = paths.concat(exvivoBaseDir, 'test' .. seq_dix .. '_labels_head.json')
    table.insert(exvivo_json_files, json_file_path)
end
--local anno_tab = readtoolLabelFile(exvivo_json_files)
--local test_anno_tab = internalSepTrainingData(anno_tab, 1)
--print(#test_anno_tab)
--print(paths.concat(exvivoBaseDir, 'test_endo_toolpos_head.t7'))
--torch.save(paths.concat(exvivoBaseDir, 'test_endo_toolpos_head.t7'), test_anno_tab)


-- ----------------------------------------------------------------------------------------------------
-- seq new instrument and old instrument
local new_exvivoBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing/tool_label'
local pure_old_exvivo_json_files = {}
for seq_dix = 1, 4 do
    local json_file_path = paths.concat(exvivoBaseDir, 'test' .. seq_dix .. '_labels_head.json')
    table.insert(pure_old_exvivo_json_files, json_file_path)
end
local pure_old_anno_tab = readtoolLabelFile(pure_old_exvivo_json_files)
local pure_old_test_anno_tab = internalSepTrainingData(pure_old_anno_tab, 1)
print('seq 1-4 old data num: ' .. #pure_old_test_anno_tab)

local new_exvivo_json_files = {}
for seq_dix = 5, 6 do
    local json_file_path = paths.concat(exvivoBaseDir, 'test' .. seq_dix .. '_labels_head.json')
    table.insert(new_exvivo_json_files, json_file_path)
end
local old_anno_tab = readOldToolLabelFile(new_exvivo_json_files)
local new_anno_tab = readNewToolLabelFile(new_exvivo_json_files)
local old_test_anno_tab = internalSepTrainingData(old_anno_tab, 1)
local new_test_anno_tab = internalSepTrainingData(new_anno_tab, 1)
print('seq 5-6 old data num: ' .. #old_test_anno_tab)
print('seq 5-6 new data num: ' .. #new_test_anno_tab)

local all_old_test_anno_tab = {}
for i=1, #pure_old_test_anno_tab do
    table.insert(all_old_test_anno_tab, pure_old_test_anno_tab[i])
end
for i=1, #old_test_anno_tab do
    table.insert(all_old_test_anno_tab, old_test_anno_tab[i])
end
print('seq 1-6 old data num: ' .. #all_old_test_anno_tab)
print(paths.concat(exvivoBaseDir, 'test_old_endo_toolpos_head.t7'))
torch.save(paths.concat(exvivoBaseDir, 'test_old_endo_toolpos_head.t7'), all_old_test_anno_tab)
print(paths.concat(exvivoBaseDir, 'test_new_endo_toolpos_head.t7'))
torch.save(paths.concat(exvivoBaseDir, 'test_new_endo_toolpos_head.t7'), new_test_anno_tab)


-- ---------------------------------------------------------------------------------------
-- -- icl data
--local iclBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/tool_label'
--local icl_json_files = {}
--for seq_idx = 2,3 do
--    local json_file_path = paths.concat(iclBaseDir, 'icl_data' .. seq_idx .. '_labels.json')
--    table.insert(icl_json_files, json_file_path)
--end
--
--local anno_tab = readtoolLabelFile(icl_json_files)
--local train_anno_tab, val_anno_tab = internalSepTrainingData(anno_tab)
--print(#train_anno_tab, #val_anno_tab)
--torch.save(paths.concat(iclBaseDir, 'train_icl_toolpos.t7'), train_anno_tab)
--torch.save(paths.concat(iclBaseDir, 'val_icl_toolpos.t7'), val_anno_tab)









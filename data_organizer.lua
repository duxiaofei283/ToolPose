require 'json'
require 'image'
require 'colormap'
require 'data_utils'
torch.setdefaulttensortype('torch.FloatTensor')

local baseDir = '/Users/xiaofeidu/workspace/sloth/tool_label'
local train_anno_tab = {}
local val_anno_tab = {}

-- read json file
-- keyword
-- filename, class=image, annotations
-- for annotations: keyword
--  class = LeftClasperPoint, RightClasperPoint HeadPoint, ShaftPoint, TrackedPoint, EndPoint.
--  id = tool1, tool2.
--  x y

for seq_idx=1, 4 do
    local jsonFilePath = paths.concat(baseDir, 'Endo' .. seq_idx .. '_labels.json')

    local json_tab = json.load(jsonFilePath)
    local frame_num = #json_tab
    local anno_tab = {}
    -- frame
    for i=1, frame_num do
        local frame_name = json_tab[i].filename
        print('frame ' .. frame_name)

        local annotations = json_tab[i].annotations
        if #annotations ~= 0 then
            table.insert(anno_tab, json_tab[i])
        end
    end

    local anno_frame_num = #anno_tab
    assert(anno_frame_num >= 1)
    local train_anno_frame_num = math.max(math.floor(0.8 * anno_frame_num), 1)

    for i=1, train_anno_frame_num do
        table.insert(train_anno_tab, anno_tab[i])
    end
    for i=train_anno_frame_num+1, anno_frame_num do
        table.insert(val_anno_tab, anno_tab[i])
    end

end

torch.save(paths.concat(baseDir, 'train_endo_toolpos.t7'), train_anno_tab)
torch.save(paths.concat(baseDir, 'val_endo_toolpos.t7'), val_anno_tab)
print('===========================================================================')
-- ---------------------------------------------------------------------------------------

-- data augmentation
-- raw image format: img_0000xx_raw.png
-- gen joint map and heat map
local radius = 10
local sigma = 20
local train_aug_anno_tab = {}
local val_aug_anno_tab = {}

local aug_data_tabs = {train_aug_anno_tab, val_aug_anno_tab}

for xx = 1, #aug_data_tabs do
    local aug_data_tab = aug_data_tabs[xx]

    for i=1, #train_anno_tab do
        local frame_anno_tab = train_anno_tab[i]
        local frame_name = frame_anno_tab.filename
        print('Augment frame ' .. frame_name)
        local frame = image.load(frame_name, 3, 'byte')
        local frame_extname = paths.extname(frame_name)
        local frame_dir = paths.dirname(frame_name)
        local aug_dir = paths.concat(paths.dirname(frame_dir), 'aug')
        if not paths.dirp(aug_dir) then
            paths.mkdir(aug_dir)
        end

        local frame_basename = paths.basename(frame_name, frame_extname)
        local frame_idx = tonumber(string.match(frame_basename, '%d+'))

        for degree = -2, 2 do  -- rotation
            for flip = 0, 1 do  -- no flip and flip
                -- augmented frame
                local aug_frame, aug_annos
                local aug_frame_name_format = 'img_%06d_f%d_d%d_raw.%s'
                local aug_frame_name = paths.concat(aug_dir, string.format(aug_frame_name_format, frame_idx, flip, degree, frame_extname))

                aug_frame, aug_annos = flipToolPosData(frame, flip, frame_anno_tab.annotations)
                aug_frame, aug_annos = rotateToolPos(aug_frame, degree, aug_annos)

                image.save(aug_frame_name, aug_frame)

                -- augmented jointmap and heatmap
                local joint_map, heat_map, jointmap_name, heatmap_name
                local jointmap_name_format = 'img_%06d_f%d_d%d_%s_%s_jointmap.%s'
                local heatmap_name_format =  'img_%06d_f%d_d%d_%s_%s_heatmap.%s'
                for j=1, #aug_annos do
                    local joint_anno = aug_annos[j]
                    local joint_class = joint_anno.class
                    local joint_id = joint_anno.id
                    print(string.format('augmenting %d, %d, %s', degree, flip, joint_class))
                    jointmap_name = paths.concat(aug_dir, string.format(jointmap_name_format, frame_idx, flip, degree, joint_id, joint_class, frame_extname))
                    heatmap_name = paths.concat(aug_dir, string.format(heatmap_name_format, frame_idx, flip, degree, joint_id, joint_class, frame_extname))

                    print(string.format('ori joint_pos: [%f, %f]', frame_anno_tab.annotations[j].x, frame_anno_tab.annotations[j].y))
                    print(string.format('aug joint_pos: [%f, %f]', joint_anno.x, joint_anno.y))
                    joint_map = genJointMap(joint_anno.x, joint_anno.y,  radius, aug_frame)
                    heat_map = genHeatMapFast(joint_anno.x, joint_anno.y, sigma, aug_frame)
                    local colormap_ = colormap:convert(heat_map)
                    image.save(jointmap_name, joint_map)
                    joint_anno.jointmapname = jointmap_name
                    image.save(heatmap_name, colormap_)
                    joint_anno.heatmapname = heatmap_name
                end

                local aug_frame_anno_tab = {}
                aug_frame_anno_tab.class = 'image'
                aug_frame_anno_tab.filename = aug_frame_name
                aug_frame_anno_tab.annotations = aug_annos

                table.insert(aug_data_tab, aug_frame_anno_tab)
            end
        end
    end

end

torch.save(paths.concat(baseDir, 'train_endo_aug_toolpos.t7'), train_aug_anno_tab)
torch.save(paths.concat(baseDir, 'val_endo_aug_toolpos.t7'), val_aug_anno_tab)




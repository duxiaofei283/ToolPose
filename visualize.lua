require 'image'
require 'paths'
require 'csvigo'
require 'distributions'
require 'colormap'
torch.setdefaulttensortype('torch.FloatTensor')

-- find the shaft end point of the instrument
local function findShaftEndPt(cx, cy, ax, ay, class_mask)
    local frame_width = class_mask:size(3)
    local frame_height = class_mask:size(2)
    local max_length = math.sqrt(math.pow(frame_height,2)+math.pow(frame_width,2))
    local border_margin = 20

    local end_pt_x, end_pt_y
    local orient = 1
    for l=0, max_length do
        -- decide the orientation
        end_pt_x = cx + orient * l * ax
        end_pt_y = cy + orient * l * ay
        if class_mask[1][torch.round(end_pt_y)][torch.round(end_pt_x)] ==  70 then -- shaft=160, manipulator=70, bg=0
            orient = -1 * orient -- change orientation
        end
        if end_pt_x <= border_margin or end_pt_x >= frame_width-border_margin or end_pt_y <= border_margin or end_pt_y >= frame_height-border_margin then
--           class_mask[1][torch.round(end_pt_y)][torch.round(end_pt_x)] ==  0 then
            break
        end
    end
    return end_pt_x, end_pt_y
end

-- gen binary joint map
local function genJointmap(ptx, pty, radius, img)
    local height = img:size(2)
    local width = img:size(3)
    local jointmap = torch.zeros(1, height, width)

    for i=torch.round(ptx-radius), torch.round(ptx+radius) do
        for j=torch.round(pty-radius), torch.round(pty+radius) do
            if math.sqrt(math.pow(i-ptx,2)+math.pow(j-pty,2)) <= radius and i>=1 and i<=width and j>=1 and j<=height then
                jointmap[1][j][i] = 1
            end
        end
    end
    return jointmap
end

-- gen regression heatmap
local function genHeatmap(ptx, pty, sigma, img)
    local mu = torch.Tensor({ptx, pty})
    local sigma_matrix = torch.eye(2)
    sigma_matrix[1][1] = sigma
    sigma_matrix[2][2] = sigma
    local height = img:size(2)
    local width = img:size(3)
    local heatmap = torch.zeros(1, height, width)
    for i=1, width do
        for j=1, height do
            local prob = distributions.mvn.pdf(torch.Tensor({i,j}), mu, sigma_matrix)
            heatmap[1][j][i] = prob
        end
    end
    return heatmap
end

local baseDir = '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Rigid_Training/Training'
local seqName = 'OP2'
local imgDir = paths.concat(baseDir, seqName,  'Raw')
local maskDir = paths.concat(baseDir, seqName, 'Masks')
local cvsFilePath = paths.concat(baseDir, seqName, 'Instruments_' .. seqName .. '_withHeader.csv')
local vis_saveDir = paths.concat(baseDir, seqName, 'Vis')


-- read csv file
-- keyword
-- imgname, ins1_center_x, ins1_center_y, ins1_axis_x, ins1_axis_y, ins2_center_x, ins2_center_y, ins2_axis_x ,ins2_axis_y

local query = csvigo.load{path=cvsFilePath, mode='query'}
local cvs_tab = query('all')
local anno_num = #cvs_tab.imgname

-- frame
for i=1, anno_num do
    local frame_name = cvs_tab.imgname[i]
    print('frame ' .. frame_name)
    local frame = image.load(paths.concat(imgDir, frame_name))

    -- instrument mask
    local frame_idx_string = string.match(frame_name, '%d+')
    local frame_idx = tonumber(frame_idx_string)
    local ins_mask_name = 'img_' .. frame_idx_string .. '_instrument.png'
    local class_mask_name = 'img_' .. frame_idx_string .. '_class.png'

    local ins_mask = image.load(paths.concat(maskDir, ins_mask_name), 1, 'byte')
    local class_mask = image.load(paths.concat(maskDir, class_mask_name), 1, 'byte')

    local ins1_cx = tonumber(cvs_tab.ins1_center_x[i])
    local ins1_cy = tonumber(cvs_tab.ins1_center_y[i])
    local ins2_cx = tonumber(cvs_tab.ins2_center_x[i])
    local ins2_cy = tonumber(cvs_tab.ins2_center_y[i])

    local ins1_ax = tonumber(cvs_tab.ins1_axis_x[i])
    local ins1_ay = tonumber(cvs_tab.ins1_axis_y[i])
    local ins2_ax = tonumber(cvs_tab.ins2_axis_x[i])
    local ins2_ay = tonumber(cvs_tab.ins2_axis_y[i])

    local frame_draw = frame:clone()
    if ins1_cx ~= nil and ins1_cy ~= nil and ins1_ax ~= nil and ins1_ax ~= nil then
        frame_draw = image.drawRect(frame_draw, ins1_cx-2, ins1_cy-2, ins1_cx+2, ins1_cy+2, {color={0,255,0}})
        -- draw end point
        local end_ptx, end_pty = findShaftEndPt(ins1_cx, ins1_cy, ins1_ax, ins1_ay, class_mask)
--        print('ins1 endpoint: ' .. end_ptx .. ',' .. end_pty)
        frame_draw = image.drawRect(frame_draw, end_ptx-2, end_pty-2, end_ptx+2, end_pty+2, {color={255,0,0}})

        -- genearte center point binarymap
        local radius = 20
        local cp_jointmap = genJointmap(ins1_cx, ins1_cy, radius, frame)
        image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_cpBimap.png'), cp_jointmap)


        -- generate center point heatmap
        local sigma = 20
        local cp_heatmap = genHeatmap(ins1_cx, ins1_cy, sigma, frame)
        local cp_colormap = colormap:convert(cp_heatmap)
        image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_cpColormap.png'), cp_colormap)

        -- genearte end point binarymap
        local ep_jointmap = genJointmap(end_ptx, end_pty, radius, frame)
        image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_epBimap.png'), ep_jointmap)

        -- generate end point heatmap
        local ep_heatmap = genHeatmap(end_ptx, end_pty, sigma, frame)
        local ep_colormap = colormap:convert(ep_heatmap)
        image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_epColormap.png'), ep_colormap)

        local frame_fused = frame * 0.7 + cp_colormap:float() * 0.15 + ep_colormap:float() * 0.15
        image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_fused.png'), frame_fused)
    end

    if ins2_cx ~= nil and ins2_cy ~= nil and ins2_ax ~= nil and ins2_ax ~= nil then
        frame_draw = image.drawRect(frame_draw, ins2_cx-2, ins2_cy-2, ins2_cx+2, ins2_cy+2, {color={255,255,0}})
        -- draw end point
        local end_ptx, end_pty = findShaftEndPt(ins2_cx, ins2_cy, ins2_ax, ins2_ay, class_mask)
        print('ins2 endpoint: ' .. end_ptx .. ',' .. end_pty)
        frame_draw = image.drawRect(frame_draw, end_ptx-2, end_pty-2, end_ptx+2, end_pty+2, {color={0,0,255}})

        -- generate centerpoint heatmap

    end

    image.save(paths.concat(vis_saveDir, 'img_' .. frame_idx_string .. '_draw.png'), frame_draw)
end







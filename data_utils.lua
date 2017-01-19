require 'image'
require 'distributions'
torch.setdefaulttensortype('torch.FloatTensor')


function point2newFileLocation(oldFileName, pattern, replace)
    local newFileName, changedFlag = string.gsub(oldFileName, pattern, replace)
    assert(changedFlag == 1)
    return newFileName
end

--function shallowCopy(src)
--    local src_type = type(src)
--    local dst
--    if src_type == 'table' then
--        dst = {}
--        for key, value in pairs(src) do
--            dst[key] = value
--        end
--    else -- number, string, boolean, etc
--        dst = src
--    end
--    return dst
--end

function deepCopy(src)
    local src_type = type(src)
    local dst
    if src_type == 'table' then
        dst = {}
        for key, value in next, src, nil do
            dst[deepCopy(key)] = deepCopy(value)
        end
    else -- number, string, boolean, etc
        dst = src
    end
    return dst
end

function preProcess(imgsRGB, inputWidth, inputHeight, meanRGBValue)
    -- convert to input image size
    local imgs_scaled
    if torch.type(imgsRGB) == 'table' then
        imgs_scaled = torch.FloatTensor(#imgsRGB, 3, inputHeight, inputWidth)
        for i=1, #imgsRGB do
            imgs_scaled[i] = image.scale(imgsRGB[i], inputWidth, inputHeight)
        end
    else
        imgs_scaled = image.scale(imgsRGB, inputWidth, inputHeight)
    end

--    -- convert RGB to BGR
--    local perm = torch.LongTensor{3, 2, 1 }
--    local imgsBGR = imgs_scaled:index(2, perm):float()

    if meanRGBValue == nil then
        meanRGBValue = torch.FloatTensor({123, 117, 104})
    end
--    -- substract mean values
--    for i=1, 3 do
--        imgs_scaled:select(2, i):add(-1*meanRGBValue[i])
--    end

    return imgs_scaled
end

-- [discard, use genJointMapNew] generate binary joint map
function genJointMap(ptx, pty, radius, frame)
    local height = frame:size(2)
    local width = frame:size(3)
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


--toolJointNames = {'LeftClasperPoint=2', 'RightClasperPoint=3',
--                          'HeadPoint=4', 'ShaftPoint=5',
--                          'TrackedPoint=6', 'EndPoint=7' } -- joint number = 6
-- background=1
-- note: annotations is normalized
function genJointMapNew(annotations, jointNames, radius, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    radius = radius / scale
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)
    local jm_height = torch.floor(frame_height / scale)
    local jm_width = torch.floor(frame_width / scale)
    local jointmap = torch.ByteTensor(1, jm_height, jm_width):fill(1)

    for joint_idx=1, #jointNames do
        local joint_anno = annotations[jointNames[joint_idx]]
        if joint_anno ~= nil then
            for tool_idx=1, #joint_anno do
                local joint_x = joint_anno[tool_idx].x * frame_width /scale
                local joint_y = joint_anno[tool_idx].y * frame_height/scale
                local x_min = math.max(1, torch.round( joint_x - radius))
                local x_max = math.min(jm_width, torch.round(joint_x + radius))
                local y_min = math.max(1, torch.round(joint_y - radius))
                local y_max = math.min(jm_height, torch.round(joint_y + radius))

                for i=x_min, x_max do
                    for j=y_min, y_max do
                        if math.sqrt(math.pow(i-joint_x, 2) + math.pow(j-joint_y, 2)) <= radius then
                            if jointmap[1][j][i] ~= 1 then
                                -- dist between new class and old_class
                                local old_class = jointmap[1][j][i]-1
                                local old_dist = 1e+8  -- large number

                                for old_tool_idx=1, #annotations[jointNames[old_class]] do
                                    local old_toolj_dist= math.sqrt(math.pow(i-annotations[jointNames[old_class]][old_tool_idx].x*frame_width/scale, 2) +
                                                                    math.pow(j-annotations[jointNames[old_class]][old_tool_idx].y*frame_height/scale, 2))
                                    if old_toolj_dist < old_dist then old_dist = old_toolj_dist end
                                end

                                local new_dist = math.sqrt(math.pow(i-joint_x, 2) + math.pow(j-joint_y, 2))
--                                print(old_class, joint_idx)
--                                print(old_dist, new_dist)
                                if new_dist < old_dist then
                                    jointmap[1][j][i] = joint_idx+1
                                end

                            else
                                jointmap[1][j][i] = joint_idx+1 -- plus 1 because of background
                            end
                        end
                    end
                end

            end
        end
    end

    return jointmap
end

function genSepJointMap(annotations, jointNames, radius, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    radius = radius / scale
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)
    local jm_height = torch.floor(frame_height / scale)
    local jm_width = torch.floor(frame_width / scale)
    local joint_num = #jointNames
    local jointmap = torch.ByteTensor(joint_num, jm_height, jm_width):fill(0)

    for joint_idx=1, #jointNames do
        local joint_anno = annotations[jointNames[joint_idx]]
        if joint_anno ~= nil then
            for tool_idx=1, #joint_anno do
                local joint_x = joint_anno[tool_idx].x * frame_width /scale
                local joint_y = joint_anno[tool_idx].y * frame_height/scale
                local x_min = math.max(1, torch.round( joint_x - radius))
                local x_max = math.min(jm_width, torch.round(joint_x + radius))
                local y_min = math.max(1, torch.round(joint_y - radius))
                local y_max = math.min(jm_height, torch.round(joint_y + radius))

                for i=x_min, x_max do
                    for j=y_min, y_max do
                        if math.sqrt(math.pow(i-joint_x, 2) + math.pow(j-joint_y, 2)) <= radius then
                            jointmap[joint_idx][j][i] = 1
                        end
                    end
                end

            end
        end
    end

    return jointmap
end

-- [discard, use genHeatMapFast] generate regression heatmap
function genHeatMap(ptx, pty, sigma, frame)
    local mu = torch.Tensor({ptx, pty})
    local sigma_matrix = torch.eye(2) * sigma
    local height = frame:size(2)
    local width = frame:size(3)
    local heatmap = torch.zeros(1, height, width)
    for i=1, width do
        for j=1, height do
            local prob = distributions.mvn.pdf(torch.Tensor({i,j}), mu, sigma_matrix)
            heatmap[1][j][i] = prob
        end
    end
    return heatmap
end

-- normalized ptx and pty
function genHeatMapFast(ptx, pty, sigma, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)

    sigma = sigma / scale
    ptx = ptx * frame_width / scale
    pty = pty * frame_height/ scale

    local mu = torch.Tensor({ptx, pty})
    local sigma_matrix = torch.eye(2) * sigma
    local hm_height = torch.floor(frame_height / scale)
    local hm_width = torch.floor(frame_width / scale)
    local heatmap = torch.zeros(1, hm_height, hm_width)

    local r = 2
    local x_min = math.max(1, torch.floor(ptx - r * sigma))
    local x_max = math.min(width, torch.ceil(ptx + r * sigma))
    local y_min = math.max(1, torch.floor(pty - r * sigma))
    local y_max = math.min(height, torch.ceil(pty + r * sigma))
    for i = x_min, x_max do
        for j= y_min, y_max do
            heatmap[1][j][i] = distributions.mvn.pdf(torch.Tensor({i,j}), mu, sigma_matrix)
        end
    end
    return heatmap
end

-- flip (for normalized annotations)
function flipToolPosData(frame, flip, annotations)
    local flipped_frame, flipped_annos
    flipped_annos = deepCopy(annotations)

    if flip == 0 then
        flipped_frame = frame:clone()
    else
        local frame_width = frame:size(3)
        flipped_frame = image.hflip(frame)

        for joint_class, __ in pairs(flipped_annos) do
            for i=1, #flipped_annos[joint_class] do
                -- x,y -> flip x, not y
                flipped_annos[joint_class][i].x = 1 - annotations[joint_class][i].x + 1/frame_width
--                flipped_annos[joint_class][i].y = annotations[joint_class][i].y

                if joint_class == 'LeftClasperPoint' then
                    -- class -> flip left and right clasper
                    flipped_annos['LeftClasperPoint'][i].x = 1 - annotations['RightClasperPoint'][i].x + 1/frame_width
                    flipped_annos['LeftClasperPoint'][i].y = annotations['RightClasperPoint'][i].y
                elseif joint_class == 'RightClasperPoint' then
                    flipped_annos['RightClasperPoint'][i].x = 1 - annotations['LeftClasperPoint'][i].x + 1/frame_width
                    flipped_annos['RightClasperPoint'][i].y = annotations['LeftClasperPoint'][i].y
                end
            end
        end
    end

    return flipped_frame, flipped_annos
end

-- rotate frame to simulate small camera roll movement (counter-clockwise)
function rotateToolPos(frame, degree, annotations)
    local rotated_annos = deepCopy(annotations)
    degree = degree % 360 -- [0, 360)
    local radian = degree * math.pi / 180 -- [0, 2pi)
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local rotated_frame = image.rotate(frame, radian, 'bilinear')

    -- rotate the position (visualize to check)
    local cx, cy = 0.5, 0.5
    for joint_class, __ in pairs(annotations) do
        for i=1, #annotations[joint_class] do
            local rx = annotations[joint_class][i].x - cx
            local ry = annotations[joint_class][i].y - cy
    --        print('rel x=' .. rx)
    --        print('rel y=' .. ry)
            local rl = math.sqrt(math.pow(rx,2)+math.pow(ry,2))
            local rad = torch.atan2(ry, rx) % (2*math.pi) -- [0, 2pi]
            local rotated_rad = (rad - radian) % (2*math.pi) -- [0, 2pi)
    --        print('rotated_degree=' .. rotated_rad)
            rx = torch.cos(rotated_rad) * rl
            ry = torch.sin(rotated_rad) * rl
    --        print('rot cos=' .. torch.cos(rotated_rad))
    --        print('rot sin=' .. torch.sin(rotated_rad))
    --        print('rot rel x=' .. rx)
    --        print('rot rel y=' .. ry)
            rotated_annos[joint_class][i].x = cx + rx
            rotated_annos[joint_class][i].y = cy + ry

            -- check if the rotated annotations are still inside the image
            if rotated_annos[joint_class][i].x * frame_width < 1 or rotated_annos[joint_class][i].x > 1 or
               rotated_annos[joint_class][i].y * frame_height< 1 or rotated_annos[joint_class][i].y > 1 then
--                print('Rotate out of frame')
--                print(rotated_annos[joint_class][i])
                rotated_annos[joint_class][i] = nil
            end
        end
    end
    return rotated_frame, rotated_annos
end

function rotateToolPosOrigin(frame, degree, annotations)
    local rotated_annos = deepCopy(annotations)
    degree = degree % 360 -- [0, 360)
    local radian = degree * math.pi / 180 -- [0, 2pi)
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local rotated_frame = image.rotate(frame, radian, 'bilinear')

    -- rotate the position (visualize to check)
    local cx = frame_width / 2
    local cy = frame_height / 2
    for joint_class, __ in pairs(annotations) do
        for i=1, #annotations[joint_class] do
            local rx = annotations[joint_class][i].x - cx
            local ry = annotations[joint_class][i].y - cy
    --        print('rel x=' .. rx)
    --        print('rel y=' .. ry)
            local rl = math.sqrt(math.pow(rx,2)+math.pow(ry,2))
            local rad = torch.atan2(ry, rx) % (2*math.pi) -- [0, 2pi]
            local rotated_rad = (rad - radian) % (2*math.pi) -- [0, 2pi)
    --        print('rotated_degree=' .. rotated_rad)
            rx = torch.cos(rotated_rad) * rl
            ry = torch.sin(rotated_rad) * rl
    --        print('rot cos=' .. torch.cos(rotated_rad))
    --        print('rot sin=' .. torch.sin(rotated_rad))
    --        print('rot rel x=' .. rx)
    --        print('rot rel y=' .. ry)
            rotated_annos[joint_class][i].x = cx + rx
            rotated_annos[joint_class][i].y = cy + ry

            -- check if the rotated annotations are still inside the image
            if rotated_annos[joint_class][i].x < 1 or rotated_annos[joint_class][i].x > frame_width or
               rotated_annos[joint_class][i].y < 1 or rotated_annos[joint_class][i].y > frame_height then
--                print('Rotate out of frame')
--                print(rotated_annos[joint_class][i])
                rotated_annos[joint_class][i] = nil
            end
        end
    end
    return rotated_frame, rotated_annos
end

-- normalize pose to (-0.5, 0.5]
function normalizeToolPosWithNeg(frame_width, frame_height, annotations)
    local normalized_annos = deepCopy(annotations)
    for joint_class, __ in pairs(normalized_annos) do
        for i=1, #normalized_annos[joint_class] do
            normalized_annos[joint_class][i].x = (annotations[joint_class][i].x - frame_width*0.5) / frame_width
            normalized_annos[joint_class][i].y = (annotations[joint_class][i].y - frame_height*0.5) / frame_height
        end
    end
    return normalized_annos
end

-- normalize pose to (0,1]
function normalizeToolPos01(frame_width, frame_height, annotations)
    local normalized_annos = deepCopy(annotations)
    for joint_class, __ in pairs(normalized_annos) do
        for i=1, #normalized_annos[joint_class] do
            normalized_annos[joint_class][i].x = annotations[joint_class][i].x / frame_width
            normalized_annos[joint_class][i].y = annotations[joint_class][i].y / frame_height
        end
    end
    return normalized_annos
end

function unNormalizeToolPos01(frame_width, frame_height, normalized_annotations)
    local annos = deepCopy(normalized_annotations)
    for joint_class, __ in pairs(annos) do
        for i=1, #annos[joint_class] do
            annos[joint_class][i].x = normalized_annotations[joint_class][i].x * frame_width
            annos[joint_class][i].y = normalized_annotations[joint_class][i].y * frame_height
        end
    end
    return annos
end


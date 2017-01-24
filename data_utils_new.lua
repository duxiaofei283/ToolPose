require 'image'
require 'distributions'
local toolCompo = require 'toolComponent'
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

function nmsBox(boxes, overlap)
    -- todo
end

-- for 2D heatmap, perform the non-maximum suppression by point-wise distanace,
-- return point position (row, column)
function nmsPt(map, dist_thres, score_factor_thres)
    assert(map:dim() == 2)
    map = map:float()
    local pick = {}
    local max_score = map:max()
    local map_height = map:size(1)
    local map_width = map:size(2)
    map[torch.lt(map, score_factor_thres*max_score)] = 0

    local map_aug = torch.zeros(map_height+2, map_width+2)
    local map_aug1 = map_aug:clone()
    local map_aug2 = map_aug:clone()
    local map_aug3 = map_aug:clone()
    local map_aug4 = map_aug:clone()
    map_aug[{{2,-2},{2,-2}}] = map:clone()
    map_aug1[{{2,-2},{1,-3}}] = map:clone()
    map_aug2[{{2,-2},{3,-1}}] = map:clone()
    map_aug3[{{1,-3},{2,-2}}] = map:clone()
    map_aug4[{{3,-1},{2,-2}}] = map:clone()

    local peak_map1 = torch.gt(map_aug, map_aug1)
    local peak_map2 = torch.gt(map_aug, map_aug2)
    local peak_map3 = torch.gt(map_aug, map_aug3)
    local peak_map4 = torch.gt(map_aug, map_aug4)

    --    peak_map1:cmul(peak_map2):cmul(peak_map3):cmul(peak_map4)
--    local peak_map = peak_map1[{{2,-2},{2,-2}}]

    local peak_map1_ = torch.ge(map_aug, map_aug1)
    local peak_map2_ = torch.ge(map_aug, map_aug2)
    local peak_map3_ = torch.ge(map_aug, map_aug3)
    local peak_map4_ = torch.ge(map_aug, map_aug4)
    peak_map1_:cmul(peak_map2):cmul(peak_map3):cmul(peak_map4)
    peak_map2_:cmul(peak_map1):cmul(peak_map3):cmul(peak_map4)
    peak_map3_:cmul(peak_map1):cmul(peak_map2):cmul(peak_map4)
    peak_map4_:cmul(peak_map1):cmul(peak_map2):cmul(peak_map3)
    peak_map1_:add(peak_map2_):add(peak_map3_):add(peak_map4_):clamp(0,1)
    local peak_map = peak_map1_[{{2,-2},{2,-2}}]

    local non_zero_indices = peak_map:nonzero()

    if non_zero_indices:nElement() == 0 then return pick end
    if non_zero_indices:size(1) > 50 then return pick end

    local peaks = torch.FloatTensor(non_zero_indices:size(1), 3)
    peaks[{{},{1,2}}] = non_zero_indices
    for i=1, peaks:size(1) do
        peaks[i][3] = map[peaks[i][1]][peaks[i][2]]
    end

    local r = peaks[{{}, 1}]
    local c = peaks[{{}, 2}]
    local score = peaks[{{}, 3}]

    local sorted_score, sorted_idx = torch.sort(score, false)
    local sorted_idx_tab = torch.totable(sorted_idx)
    while #sorted_idx_tab ~= 0 do
        local last = #sorted_idx_tab
        local max_idx = sorted_idx_tab[last]
        table.insert(pick, torch.totable(peaks[max_idx]))
        local removed_indice = {}
        table.insert(removed_indice, last)
        for idx=last-1, 1, -1 do
            local sorted_idx = sorted_idx_tab[idx]
            -- compute distance
            local dist = math.sqrt(math.pow(r[max_idx]-r[sorted_idx],2)+math.pow(c[max_idx]-c[sorted_idx],2))
            if dist < dist_thres then
                table.insert(removed_indice, idx)
            end
        end
        for ridx=1, #removed_indice do
--                print('removing ' .. removed_indice[r])
            table.remove(sorted_idx_tab, removed_indice[ridx])
        end
--            print('size = ' .. #sorted_idx_tab)
--            print(sorted_idx_tab)
    end
    return pick






--    local weighted_pts = torch.FloatTensor(map:nElement(),3)
--    local picked_num = 0
--    for i=1, map_height do
--        for j=1, map_width do
--            if map[i][j] >= max_score * score_factor_thres then
--                picked_num = picked_num + 1
--                weighted_pts[picked_num] = torch.FloatTensor({i, j, map[i][j]})
--            end
--        end
--    end
--
--    weighted_pts = weighted_pts[{{1, picked_num}}]
--    if weighted_pts == nil or weighted_pts:size(1) == 0 then
--    else
--        local x = weighted_pts[{{}, 1}]
--        local y = weighted_pts[{{}, 2}]
--        local score = weighted_pts[{{}, 3}]
--
--        local sorted_score, sorted_idx = torch.sort(score, false)
--        local sorted_idx_tab = torch.totable(sorted_idx)
--        while #sorted_idx_tab ~= 0 do
--            local last = #sorted_idx_tab
--            local max_idx = sorted_idx_tab[last]
--            table.insert(pick, torch.totable(weighted_pts[max_idx]))
--            local removed_indice = {}
--            table.insert(removed_indice, last)
--            for idx=last-1, 1, -1 do
--                local sorted_idx = sorted_idx_tab[idx]
--                -- compute distance
--                local dist = math.sqrt(math.pow(x[max_idx]-x[sorted_idx],2)+math.pow(y[max_idx]-y[sorted_idx],2))
--                if dist < dist_thres then
--                    table.insert(removed_indice, idx)
--                end
--            end
--            for r=1, #removed_indice do
----                print('removing ' .. removed_indice[r])
--                table.remove(sorted_idx_tab, removed_indice[r])
--            end
----            print('size = ' .. #sorted_idx_tab)
----            print(sorted_idx_tab)
--        end
--    end
----    print(pick)
--    return pick
----    return torch.FloatTensor(pick)
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

--    -- substract mean values
--    for i=1, 3 do
--        imgs_scaled:select(2, i):add(-1*meanRGBValue[i])
--    end

    -- normailize
    imgs_scaled = torch.div(imgs_scaled, 255)
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


function getJointPos(annotations, jointNames, joint_num)
    local joint_num = #jointNames
    local jointPos = torch.FloatTensor(joint_num*joint_num, 2):fill(0)
    for joint_idx = 1, #jointNames do
        local joint_anno = annotations[jointNames[joint_idx]]
        if joint_anno ~= nil then
            for tool_idx=1, #joint_anno do
                local joint_x = joint_anno[tool_idx].x
                local joint_y = joint_anno[tool_idx].y
                jointPos[joint_idx*2-1][1] = joint_x
                jointPos[joint_idx*2][2] = joint_y
            end
        end
    end

    return jointPos
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

-- generate seperate joint map
-- todo: not consider same component from different tools overlapping with each other
function genSepJointMap(annotations, jointNames, radius, frame, scale)
    if scale == nil or scale == 0 then scale = 1 end
    radius = radius / scale
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)
    local jm_height = torch.floor(frame_height / scale)
    local jm_width = torch.floor(frame_width / scale)
    local joint_num = #jointNames
    local jointmap = torch.ByteTensor(joint_num, jm_height, jm_width):fill(0)

    for joint_idx=1, joint_num do
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


-- generate seperate PAF map
-- toolCompoNames = {LeftClasper    ={LeftClasperPoint, HeadPoint},
--                   RightClasper   ={RightClasperPoint, HeadPoint},
--                   Head           ={HeadPoint, ShaftPoint},
--                   Shaft          ={ShaftPoint, EndPoint} }
function genPAFMap(annotations, toolCompoNames, side_thickness, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    side_thickness =  side_thickness/scale
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local pm_width = torch.floor(frame_width/scale)
    local pm_height = torch.floor(frame_height/scale)
    local compo_num=#toolCompoNames
    local paf_map = torch.FloatTensor(2, pm_height, pm_width):fill(0)

    for compo_idx = 1, compo_num do
        local compo = toolCompoNames[compo_idx]
        local joint1_anno = annotations[compo[1]]
        local joint2_anno = annotations[compo[2]]
        if joint1_anno == nil or joint2_anno == nil then
        else
            local paf_overlap_map = torch.FloatTensor(pm_height, pm_width):fill(1)
            -- multi tools
            for joint1_tool_idx=1, #joint1_anno do
                for joint2_tool_idx=1, #joint2_anno do
                    if joint2_anno[joint2_tool_idx].id == joint1_anno[joint1_tool_idx].id then
                        local joint1_x = joint1_anno[joint1_tool_idx].x * frame_width/scale
                        local joint1_y = joint1_anno[joint2_tool_idx].y * frame_height/scale
                        local joint2_x = joint2_anno[joint2_tool_idx].x * frame_width/scale
                        local joint2_y = joint2_anno[joint2_tool_idx].y * frame_height/scale

                        -- generate paf points map
                        local tool_comp = toolCompo(joint1_x, joint1_y, joint2_x, joint2_y, side_thickness)
                        local min_vertex, max_vertex = tool_comp:getBoundingVertices()
                        local x_min = math.max(1, torch.round( min_vertex.x))
                        local x_max = math.min(pm_width, torch.round(max_vertex.x))
                        local y_min = math.max(1, torch.round(min_vertex.y))
                        local y_max = math.min(pm_height, torch.round(max_vertex.y))

                        for i=x_min, x_max do
                            for j=y_min, y_max do
                                if tool_comp:inside(i,j) then
                                    paf_map[1][j][i] = paf_map[1][j][i] + 1 * tool_comp.rad
                                    paf_map[2][j][i] = paf_map[2][j][i] + 1 * tool_comp.rad
                                    paf_overlap_map[j][i] = paf_overlap_map[j][i] + 1
                                end
                            end
                        end

                    end
                end
            end
            paf_map[1]:cdiv(paf_overlap_map)
            paf_map[2]:cdiv(paf_overlap_map)
        end
    end
    return paf_map
end

-- todo: not consider same component from different tools overlapping with each other
function genSepPAFMapDet(annotations, toolCompoNames, side_thickness, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    side_thickness =  side_thickness/scale
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local pm_width = torch.floor(frame_width/scale)
    local pm_height = torch.floor(frame_height/scale)
    local compo_num=0
    local compo_num = #toolCompoNames
    local paf_map = torch.ByteTensor(compo_num, pm_height, pm_width):fill(0)

    local compo_idx = 1
    for compo_idx=1,  compo_num do
        local compo = toolCompoNames[compo_idx]
        local joint1_anno = annotations[compo[1]]
        local joint2_anno = annotations[compo[2]]
        if joint1_anno == nil or joint2_anno == nil then
        else
            -- multi tools
            for joint1_tool_idx=1, #joint1_anno do
                for joint2_tool_idx=1, #joint2_anno do
                    if joint2_anno[joint2_tool_idx].id == joint1_anno[joint1_tool_idx].id then
                        local joint1_x = joint1_anno[joint1_tool_idx].x * frame_width/scale
                        local joint1_y = joint1_anno[joint2_tool_idx].y * frame_height/scale
                        local joint2_x = joint2_anno[joint2_tool_idx].x * frame_width/scale
                        local joint2_y = joint2_anno[joint2_tool_idx].y * frame_height/scale

                        -- generate paf points map
                        local tool_comp = toolCompo(joint1_x, joint1_y, joint2_x, joint2_y, side_thickness)
                        local min_vertex, max_vertex = tool_comp:getBoundingVertices()
                        local x_min = math.max(1, torch.round( min_vertex.x))
                        local x_max = math.min(pm_width, torch.round(max_vertex.x))
                        local y_min = math.max(1, torch.round(min_vertex.y))
                        local y_max = math.min(pm_height, torch.round(max_vertex.y))

                        for i=x_min, x_max do
                            for j=y_min, y_max do
                                if tool_comp:inside(i, j) then
                                    -- version 1
--                                    paf_map[compo_idx*2-1][j][i] = paf_map[compo_idx*2-1][j][i] + 1*torch.cos(tool_comp.rad)
--                                    paf_map[compo_idx*2][j][i] = paf_map[compo_idx*2][j][i] + 1*torch.sin(tool_comp.rad)

                                    -- version 2
                                    paf_map[compo_idx][j][i] = 1
                                end
                            end
                        end
                    end
                end
            end
        end
        compo_idx = compo_idx + 1
    end
    return paf_map
end

function genSepPAFMapReg(annotations, toolCompoNames, side_thickness, det_side_thickness, frame, scale, normalise_scale)
    if scale == nil or scale == 0 then scale = 1 end
    if normalise_scale== nil or normalise_scale == 0 then normalise_scale = 1 end
    side_thickness =  side_thickness/scale
    det_side_thickness = det_side_thickness/scale
    local min_side_thickness = math.min(side_thickness, det_side_thickness)
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local pm_width = torch.floor(frame_width/scale)
    local pm_height = torch.floor(frame_height/scale)
    local compo_num=0
    local compo_num = #toolCompoNames
    local paf_map = torch.FloatTensor(compo_num, pm_height, pm_width):fill(0)

    local compo_idx = 1
    for compo_idx=1,  compo_num do
        local compo = toolCompoNames[compo_idx]
        local joint1_anno = annotations[compo[1]]
        local joint2_anno = annotations[compo[2]]
        if joint1_anno == nil or joint2_anno == nil then
        else
            -- multi tools
            for joint1_tool_idx=1, #joint1_anno do
                for joint2_tool_idx=1, #joint2_anno do
                    if joint2_anno[joint2_tool_idx].id == joint1_anno[joint1_tool_idx].id then
                        local joint1_x = joint1_anno[joint1_tool_idx].x * frame_width/scale
                        local joint1_y = joint1_anno[joint2_tool_idx].y * frame_height/scale
                        local joint2_x = joint2_anno[joint2_tool_idx].x * frame_width/scale
                        local joint2_y = joint2_anno[joint2_tool_idx].y * frame_height/scale

                        -- generate paf points map
                        local tool_comp = toolCompo(joint1_x, joint1_y, joint2_x, joint2_y, min_side_thickness)
                        local min_vertex, max_vertex = tool_comp:getBoundingVertices()
                        local x_min = math.max(1, torch.round( min_vertex.x))
                        local x_max = math.min(pm_width, torch.round(max_vertex.x))
                        local y_min = math.max(1, torch.round(min_vertex.y))
                        local y_max = math.min(pm_height, torch.round(max_vertex.y))

                        for i=x_min, x_max do
                            for j=y_min, y_max do
                                if tool_comp:inside(i, j) then
                                    -- version 1
--                                    paf_map[compo_idx*2-1][j][i] = paf_map[compo_idx*2-1][j][i] + 1*torch.cos(tool_comp.rad)
--                                    paf_map[compo_idx*2][j][i] = paf_map[compo_idx*2][j][i] + 1*torch.sin(tool_comp.rad)

                                    -- version 2
                                    local tool_dist = tool_comp:getToolDist(i, j)
                                    local v = distributions.norm.pdf(tool_dist, 0, side_thickness)
--                                    print(string.format('tool dist=%f, gauss_v=%f, side_thickness=%f', tool_dist, v, side_thickness))
                                    paf_map[compo_idx][j][i] = v
                                end
                            end
                        end
                    end
                end
            end
        end
        compo_idx = compo_idx + 1
    end

    -- normalize?
    paf_map = normalise_scale * torch.div(paf_map, paf_map:max())
    return paf_map
end

-- [not used] generate seperate PAF map
-- toolCompoNames = {{LeftClasperPoint, HeadPoint},
--                   {RightClasperPoint, HeadPoint},
--                   {HeadPoint, ShaftPoint},
--                   {ShaftPoint, EndPoint} }
function genSepPAFMap(annotations, toolCompoNames, side_thickness, frame, scale)
    if scale == nil or scale == 0 then
        scale = 1
    end
    side_thickness =  side_thickness/scale
    local frame_width = frame:size(3)
    local frame_height = frame:size(2)
    local pm_width = torch.floor(frame_width/scale)
    local pm_height = torch.floor(frame_height/scale)
    local compo_num=0
    local compo_num = #toolCompoNames
    local paf_map = torch.FloatTensor(compo_num*2, pm_height, pm_width):fill(0)

    local compo_idx = 1
    for compo_idx=1,  compo_num do
        local compo = toolCompoNames[compo_idx]
        local joint1_anno = annotations[compo[1]]
        local joint2_anno = annotations[compo[2]]
        if joint1_anno == nil or joint2_anno == nil then
        else
            local paf_overlap_map = torch.FloatTensor(pm_height, pm_width):fill(0)
            -- multi tools
            for joint1_tool_idx=1, #joint1_anno do
                for joint2_tool_idx=1, #joint2_anno do
                    if joint2_anno[joint2_tool_idx].id == joint1_anno[joint1_tool_idx].id then
                        local joint1_x = joint1_anno[joint1_tool_idx].x * frame_width/scale
                        local joint1_y = joint1_anno[joint2_tool_idx].y * frame_height/scale
                        local joint2_x = joint2_anno[joint2_tool_idx].x * frame_width/scale
                        local joint2_y = joint2_anno[joint2_tool_idx].y * frame_height/scale

                        -- generate paf points map
                        local tool_comp = toolCompo(joint1_x, joint1_y, joint2_x, joint2_y, side_thickness)
                        local min_vertex, max_vertex = tool_comp:getBoundingVertices()
                        local x_min = math.max(1, torch.round( min_vertex.x))
                        local x_max = math.min(pm_width, torch.round(max_vertex.x))
                        local y_min = math.max(1, torch.round(min_vertex.y))
                        local y_max = math.min(pm_height, torch.round(max_vertex.y))

                        for i=x_min, x_max do
                            for j=y_min, y_max do
                                if tool_comp:inside(i, j) then
                                    -- version 1
--                                    paf_map[compo_idx*2-1][j][i] = paf_map[compo_idx*2-1][j][i] + 1*torch.cos(tool_comp.rad)
--                                    paf_map[compo_idx*2][j][i] = paf_map[compo_idx*2][j][i] + 1*torch.sin(tool_comp.rad)

                                    -- version 2
                                    paf_map[compo_idx*2-1][j][i] = paf_map[compo_idx*2-1][j][i] + 1
                                    local rad = tool_comp.rad/(2*math.pi) + 0.5   -- from [-pi,pi] to [0,1]
                                    paf_map[compo_idx*2][j][i] = paf_map[compo_idx*2][j][i] + rad

                                    paf_overlap_map[j][i] = paf_overlap_map[j][i] + 1
                                end
                            end
                        end
                    end
                end
            end
            paf_overlap_map[torch.eq(paf_overlap_map,0)] = 1
            paf_map[compo_idx*2-1]:cdiv(paf_overlap_map)
            paf_map[compo_idx*2]:cdiv(paf_overlap_map)
        end
        compo_idx = compo_idx + 1
    end
    return paf_map
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
function genSepHeatMap(annotations, jointNames, sigma, det_radius, frame, scale, normalise_scale)
    if scale == nil or scale == 0 then scale = 1 end
    if normalise_scale== nil or normalise_scale == 0 then normalise_scale = 1 end
    sigma = sigma / scale
    det_radius = det_radius / scale
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)
    local hm_height = torch.floor(frame_height/scale)
    local hm_width = torch.floor(frame_width/scale)
    local joint_num = #jointNames
    local heatmap = torch.FloatTensor(joint_num, hm_height, hm_width):fill(0)
    local r = 2
    local range = math.min(det_radius, r * sigma)
    local sigma_matrix = torch.eye(2) * sigma

    local x_min, x_max, y_min, y_max
    for joint_idx=1, joint_num do
        local joint_anno = annotations[jointNames[joint_idx]]
        if joint_anno ~= nil then
            for tool_idx=1, #joint_anno do
                local joint_x = joint_anno[tool_idx].x * frame_width /scale
                local joint_y = joint_anno[tool_idx].y * frame_height/scale
                local mu = torch.FloatTensor({joint_x, joint_y})

                x_min = math.max(1, torch.floor(joint_x - range))
                x_max = math.min(hm_width, torch.ceil(joint_x + range))
                y_min = math.max(1, torch.floor(joint_y - range))
                y_max = math.min(hm_height, torch.ceil(joint_y + range))

                for i=x_min, x_max do
                    for j=y_min, y_max do
                        if math.sqrt(math.pow(i-joint_x, 2) + math.pow(j-joint_y, 2)) <= r * sigma then
                            local v = distributions.mvn.pdf(torch.Tensor({i,j}), mu, sigma_matrix)
                            heatmap[joint_idx][j][i] = math.max(heatmap[joint_idx][j][i], v)
                        end
                    end
                end
            end
        end
    end

    -- normalize?
    heatmap = normalise_scale * torch.div(heatmap, heatmap:max())

    return heatmap
end

-- normalized ptx and pty
function genHeatMapFast(ptx, pty, sigma, frame, scale)
    if scale == nil or scale == 0 then scale = 1 end
    local frame_height = frame:size(2)
    local frame_width = frame:size(3)

    sigma = sigma / scale
    ptx = ptx * frame_width / scale
    pty = pty * frame_height/ scale

    local mu = torch.FloatTensor({ptx, pty})
    local sigma_matrix = torch.eye(2) * sigma
    local hm_height = torch.floor(frame_height / scale)
    local hm_width = torch.floor(frame_width / scale)
    local heatmap = torch.zeros(1, hm_height, hm_width)

    local r = 2
    local x_min = math.max(1, torch.floor(ptx - r * sigma))
    local x_max = math.min(hm_width, torch.ceil(ptx + r * sigma))
    local y_min = math.max(1, torch.floor(pty - r * sigma))
    local y_max = math.min(hm_height, torch.ceil(pty + r * sigma))
    for i = x_min, x_max do
        for j= y_min, y_max do
            if math.sqrt(math.pow(i-ptx, 2) + math.pow(j-pty, 2)) <= r * sigma then
                heatmap[1][j][i] = distributions.mvn.pdf(torch.Tensor({i,j}), mu, sigma_matrix)
            end
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
            local rad = torch.atan2(ry, rx) % (2*math.pi) -- [0, 2pi)
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


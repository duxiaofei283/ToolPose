
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

-- toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
--                          'HeadPoint', 'ShaftPoint',
--                          'TrackedPoint', 'EndPoint' } -- joint number = 6

-- output size is 4 times smaller




-- 2 as base
local function Tool_sep4_v1(input_channels, first_layer_channels, joint_num, comp_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.SpatialBatchNormalization(first_layer_channels))
    m:add(nn.ReLU())

    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample
    local input_c, output_c, dual_c
    for i=1, 4 do
        -- select first element
        input_c = first_layer_channels * math.pow(2, i-1)
        dual_c = input_c
        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialMaxPooling(2,2,2,2))
                    :add(nn.SpatialConvolution(input_c, dual_c,3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
               )
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialMaxPooling(2,2,2,2))
                    :add(nn.SpatialConvolution(input_c, dual_c,3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
               )

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential():add(dual):add(nn.JoinTable(2)))
        for j=3, i do
            cat:add(nn.SelectTable(j-2))
        end
        m:add(cat)
    end

    -- up sample
    for i=4, 3, -1 do
        input_c = first_layer_channels * math.pow(2,i)
        output_c = first_layer_channels * math.pow(2,i-1)
        dual_c = first_layer_channels * math.pow(2, i-2)

        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
--                    :add(nn.SpatialUpSamplingBilinear(2))
--                    :add(nn.SpatialConvolution(input_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )
        dual:add(nn.Sequential()
--                    :add(nn.SpatialUpSamplingBilinear(2))
--                    :add(nn.SpatialConvolution(input_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )

        local p = nn.ParallelTable()
        p:add(nn.Sequential():add(dual):add(nn.JoinTable(2)))
        p:add(nn.Identity())

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(p):add(nn.JoinTable(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
               )
        for j=3, i+1-2 do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end
    m:add(nn.SelectTable(1))

    -- joint and comp layer
    local dual = nn.ConcatTable()
    dual:add(nn.SpatialConvolution(output_c, joint_num, 1,1))
    dual:add(nn.SpatialConvolution(output_c, comp_num*2, 1,1))
    m:add(dual):add(nn.JoinTable(2)):add(nn.Sigmoid())
--    m:add(dual):add(nn.JoinTable(2))

    return m
end


-- 2 as base
local function Tool_sep4_det(input_channels, first_layer_channels, joint_num, comp_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.SpatialBatchNormalization(first_layer_channels))
    m:add(nn.ReLU())

    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample
    local input_c, output_c, dual_c
    for i=1, 4 do
        -- select first element
        input_c = first_layer_channels * math.pow(2, i-1)
        dual_c = input_c
        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialMaxPooling(2,2,2,2))
                    :add(nn.SpatialConvolution(input_c, dual_c,3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
               )
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialMaxPooling(2,2,2,2))
                    :add(nn.SpatialConvolution(input_c, dual_c,3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
               )

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential():add(dual):add(nn.JoinTable(2)))
        for j=3, i do
            cat:add(nn.SelectTable(j-2))
        end
        m:add(cat)
    end

    -- up sample
    for i=4, 3, -1 do
        input_c = first_layer_channels * math.pow(2,i)
        output_c = first_layer_channels * math.pow(2,i-1)
        dual_c = first_layer_channels * math.pow(2, i-2)

        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
--                    :add(nn.SpatialUpSamplingBilinear(2))
--                    :add(nn.SpatialConvolution(input_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )
        dual:add(nn.Sequential()
--                    :add(nn.SpatialUpSamplingBilinear(2))
--                    :add(nn.SpatialConvolution(input_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )

        local p = nn.ParallelTable()
        p:add(nn.Sequential():add(dual):add(nn.JoinTable(2)))
        p:add(nn.Identity())

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(p):add(nn.JoinTable(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
               )
        for j=3, i+1-2 do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end
    m:add(nn.SelectTable(1))

    -- joint and comp layer
    local dual = nn.ConcatTable()
    dual:add(nn.SpatialConvolution(output_c, joint_num, 1,1))
    dual:add(nn.SpatialConvolution(output_c, comp_num, 1,1))
    m:add(dual):add(nn.JoinTable(2)):add(nn.Sigmoid())
--    m:add(dual):add(nn.JoinTable(2))

    return m
end

local dual_net = Tool_sep4_det(3, 64, 5, 4)     -- 64,128,256,512,1024
print(dual_net)
local saveDir = '/home/xiaofei/workspace/toolPose/models'
local modelConf = {type='toolDualPoseSep', v=1 }
--
local saveID = modelConf.type .. '_v' .. modelConf.v
local initModelPath = paths.concat(saveDir, 'model.' .. saveID .. '.init.t7')
torch.save(paths.concat(saveDir, initModelPath), dual_net)
print('saved model ' .. saveID .. ' to ' .. paths.concat(saveDir, initModelPath))



-- toolPoseSep v=1: j_radius = 20, inputsize=[384,480], model_output_scale=4, with rotation augment, no flip
-- toolDualPoseSep v=1: j_radius=20, inputsize=[384, 480], model_output_scale=4, with rotation augment, no flip
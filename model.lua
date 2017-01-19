
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

-- toolJointNames = {'LeftClasperPoint', 'RightClasperPoint',
--                          'HeadPoint', 'ShaftPoint',
--                          'TrackedPoint', 'EndPoint' } -- joint number = 6

local function UnetRetina(input_channels, first_layer_channels, joint_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.ReLU())

    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample
    for i=1, 4 do
        -- select first element
        local input_c = first_layer_channels * math.pow(2,i-1)
        local output_c = first_layer_channels * math.pow(2, i)
        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.ReLU())
                    :add(nn.SpatialMaxPooling(2,2,2,2))
             )
        for j=1, i do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end

    -- up sample
    for i=4, 1, -1 do
        local input_c = first_layer_channels * math.pow(2,i)
        local output_c = first_layer_channels * math.pow(2, i-1)

        local p = nn.ParallelTable()
        p:add(nn.Sequential()
                    :add(nn.SpatialUpSamplingBilinear(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
             )
        p:add(nn.Identity())

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(p):add(nn.JoinTable(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
               )

        for j=3, i+1 do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end
    m:add(nn.SelectTable(1))

    -- joint layer
    m:add(nn.SpatialConvolution(first_layer_channels, joint_num, 1, 1))

    return m
end

-- [discard] output size is 4 times smaller
local function Unet_Experiment(input_channels, first_layer_channels, joint_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.ReLU())

    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample
    local input_c, output_c
    for i=1, 4 do
        -- select first element
        input_c = first_layer_channels * math.pow(2,i-1)
        output_c = first_layer_channels * math.pow(2, i)
        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
                    :add(nn.SpatialMaxPooling(2,2,2,2))
             )
        for j=1, i do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end

    -- up sample
    for i=4, 3, -1 do
        input_c = first_layer_channels * math.pow(2,i)
        output_c = first_layer_channels * math.pow(2, i-1)

        local p = nn.ParallelTable()
        p:add(nn.Sequential()
                    :add(nn.SpatialUpSamplingBilinear(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
             )
        p:add(nn.Identity())

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(p):add(nn.JoinTable(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
               )

        for j=3, i+1 do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end
    m:add(nn.SelectTable(1))

    -- joint layer
    m:add(nn.SpatialConvolution(output_c, joint_num, 1, 1))
     :add(nn.SpatialBatchNormalization(joint_num))
     :add(nn.ReLU())


    -- permute(1,3,4,2)
    m:add(nn.Transpose({2,3}, {3,4}))

    return m
end


local function Unet_Experiment2(input_channels, first_layer_channels, joint_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.ReLU())

    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample
    local input_c, output_c
    for i=1, 4 do
        -- select first element
        input_c = first_layer_channels * math.pow(2,i-1)
        output_c = first_layer_channels * math.pow(2, i)
        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialMaxPooling(2,2,2,2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())

             )
        for j=1, i do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end

    -- up sample
    for i=4, 3, -1 do
        input_c = first_layer_channels * math.pow(2,i)
        output_c = first_layer_channels * math.pow(2, i-1)

        local p = nn.ParallelTable()
        p:add(nn.Sequential()
                    :add(nn.SpatialUpSamplingBilinear(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
             )
        p:add(nn.Identity())

        local cat = nn.ConcatTable()
        cat:add(nn.Sequential()
                    :add(nn.NarrowTable(1,2))
                    :add(p):add(nn.JoinTable(2))
                    :add(nn.SpatialConvolution(input_c, output_c, 3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(output_c))
                    :add(nn.ReLU())
               )

        for j=3, i+1 do
            cat:add(nn.SelectTable(j))
        end
        m:add(cat)
    end
    m:add(nn.SelectTable(1))

    -- joint layer
    m:add(nn.SpatialConvolution(output_c, joint_num, 1, 1))
     :add(nn.SpatialBatchNormalization(joint_num))
     :add(nn.ReLU())


    -- permute(1,3,4,2)
    m:add(nn.Transpose({2,3}, {3,4}))
    m:add(nn.View(-1, 7))
    return m
end

local function regressNet(joint_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(joint_num, 64, 9, 9, 1, 1, 4, 4))
    m:add(nn.SpatialBatchNormalization(64))
    m:add(nn.ReLU())
    -- conv2
    m:add(nn.SpatialConvolution(64, 64, 13, 13, 1, 1, 6, 6))
    m:add(nn.SpatialBatchNormalization(64))
    m:add(nn.ReLU())
    -- conv3
    m:add(nn.SpatialConvolution(64, 128, 13, 13, 1, 1, 6, 6))
    m:add(nn.SpatialBatchNormalization(128))
    m:add(nn.ReLU())
    -- conv4
    m:add(nn.SpatialConvolution(128, 256, 15, 15, 1, 1, 7, 7))
    m:add(nn.SpatialBatchNormalization(256))
    m:add(nn.ReLU())

    -- conv5
    m:add(nn.SpatialConvolution(256, 512, 1, 1))
    m:add(nn.SpatialBatchNormalization(512))
    m:add(nn.ReLU())
    -- conv6
    m:add(nn.SpatialConvolution(512, 512, 1, 1))
    m:add(nn.SpatialBatchNormalization(512))
    m:add(nn.ReLU())
    -- conv7
    m:add(nn.SpatialConvolution(512, joint_num, 1, 1))
    m:add(nn.SpatialBatchNormalization(joint_num))
    m:add(nn.ReLU())
    -- conv8
    m:add(nn.SpatialFullConvolution(joint_num, joint_num, 8, 8, 4, 4, 2, 2))
    m:add(nn.SpatialBatchNormalization(joint_num))
    m:add(nn.ReLU())

    return m
end

--unet = UnetRetina(3, 64, 7)
local joint_net = Unet_Experiment2(3, 64, 7)
local heat_net = regressNet(7)

local saveDir = '/home/xiaofei/workspace/toolPose/models'
local modelConf = {type='toolPose', v=2 }

local saveID = modelConf.type .. '_v' .. modelConf.v
local initModelPath = paths.concat(saveDir, 'model.' .. saveID .. '.init.t7')
torch.save(paths.concat(saveDir, initModelPath), joint_net)

-- v=1: j_radius = 20, inputsize[576,720], model_output_scale=4, no rotation augment, no flip
-- v=2: j_radius = 20, inputsize[384,480], model_output_scale=4, with rotation augment, no flip
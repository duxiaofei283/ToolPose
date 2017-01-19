require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

 local function ToolDetModel(input_channels, first_layer_channels, joint_num, compo_num)
    local m = nn.Sequential()
    -- conv1
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
    m:add(nn.SpatialBatchNormalization(first_layer_channels))
    m:add(nn.ReLU())
    m:add(nn.ConcatTable():add(nn.Identity()))
    -- down sample (without using pooling operations)
    local input_c, output_c, dual_c
    for i=1, 4 do
        -- select first element
        input_c = first_layer_channels * math.pow(2, i-1)
        dual_c = input_c
        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialConvolution(input_c, dual_c, 2,2,2,2,0,0))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                    :add(nn.SpatialConvolution(dual_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )
        dual:add(nn.Sequential()
                    :add(nn.SelectTable(1))
                    :add(nn.SpatialConvolution(input_c, dual_c, 2,2,2,2,0,0))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                    :add(nn.SpatialConvolution(dual_c, dual_c, 3,3,1,1,1,1))
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
    -- upsample
    for i=4, 3, -1 do
        input_c = first_layer_channels * math.pow(2,i)
        output_c = first_layer_channels * math.pow(2,i-1)
        dual_c = first_layer_channels * math.pow(2, i-2)

        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                    :add(nn.SpatialFullConvolution(dual_c, dual_c, 3,3,1,1,1,1))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                )
        dual:add(nn.Sequential()
                    :add(nn.SpatialFullConvolution(input_c, dual_c, 2,2,2,2))
                    :add(nn.SpatialBatchNormalization(dual_c))
                    :add(nn.ReLU())
                    :add(nn.SpatialFullConvolution(dual_c, dual_c, 3,3,1,1,1,1))
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
--    -- version 1
--    dual:add(nn.Sequential()
--                :add(nn.SpatialConvolution(output_c, joint_num, 1,1))
--                :add(nn.SpatialBatchNormalization(joint_num))
--                :add(nn.ReLU())
--            )
--    dual:add(nn.Sequential()
--                :add(nn.SpatialConvolution(output_c, compo_num, 1,1))
--                :add(nn.SpatialBatchNormalization(compo_num))
--                :add(nn.ReLU())
--            )
--    m:add(dual):add(nn.JoinTable(2)):add(nn.Sigmoid())

    -- version 2
    print('waweweae')
    print(input_c)
    print(output_c)
    dual:add(nn.Sequential()
                :add(nn.SpatialConvolution(output_c, dual_c, 1,1))
                :add(nn.SpatialBatchNormalization(dual_c))
                :add(nn.ReLU())
            )
    dual:add(nn.Sequential()
                :add(nn.SpatialConvolution(output_c, dual_c, 1,1))
                :add(nn.SpatialBatchNormalization(dual_c))
                :add(nn.ReLU())
            )
    m:add(dual):add(nn.JoinTable(2))
    m:add(nn.SpatialConvolution(output_c, joint_num+compo_num, 1,1))
     :add(nn.SpatialBatchNormalization(joint_num+compo_num))

    m:add(nn.Sigmoid())

    return m
end

local tool_detnet = ToolDetModel(3, 64, 5, 4)
print(tool_detnet)
local saveDir = '/home/xiaofei/workspace/toolPose/models'
local modelConf = {type='toolPartDet', v=1 }
--
local saveID = modelConf.type .. '_v' .. modelConf.v
local initModelPath = paths.concat(saveDir, 'model.' .. saveID .. '.init.t7')
torch.save(paths.concat(saveDir, initModelPath), tool_detnet)
print('saved model ' .. saveID .. ' to ' .. paths.concat(saveDir, initModelPath))



-- toolPartDet v=1: j_radius=10, inputsize=[3, 384, 480], outputsize=[5+4, 96, 120], model_output_scale=4,



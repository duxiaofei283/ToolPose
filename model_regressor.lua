require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

local function tool_regress(input_channels, first_layer_channels)
    local m = nn.Sequential()
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 9, 9, 1, 1, 4, 4))
--     :add(nn.SpatialBatchNormalization(first_layer_channels))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels, first_layer_channels*2, 13, 13, 1, 1, 6, 6))
--     :add(nn.SpatialBatchNormalization(first_layer_channels*2))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*2, first_layer_channels*4, 15,15,1,1,7,7))
--     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, first_layer_channels*4, 1,1,1,1))
     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, first_layer_channels*4, 1, 1, 1,1))
--     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, input_channels, 1, 1, 1, 1))
--     :add(nn.SpatialBatchNormalization(input_channels))
     :add(nn.ReLU())
--    m:add(nn.Sigmoid())

    return m
end

local joint_num = 5
local compo_num = 4
local regress_net = tool_regress(joint_num+compo_num, 64)
print(regress_net)
local saveDir = '/home/xiaofei/workspace/toolPose/models'
local modelConf = {type='toolPoseRegress', v=1 }
--
local saveID = modelConf.type .. '_v' .. modelConf.v
local initModelPath = paths.concat(saveDir, 'model.' .. saveID .. '.init.t7')
torch.save(paths.concat(saveDir, initModelPath), regress_net)
print('saved model ' .. saveID .. ' to ' .. paths.concat(saveDir, initModelPath))

-- toolPoseRegress v=1: inputsize=[384,480]



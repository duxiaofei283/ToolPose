require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
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
--     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
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

local function tool_regressFull(joint_num, compo_num, first_layer_channels)

    local function Dual(in_c, dual_c1, dual_c2, kernel_size, stride, pad)
        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SpatialConvolution(in_c, dual_c1, kernel_size, kernel_size, stride, stride, pad, pad))
                    :add(nn.SpatialBatchNormalization(dual_c1))
                    :add(nn.ReLU())
                )
        dual:add(nn.Sequential()
                    :add(nn.SpatialConvolution(in_c, dual_c2, kernel_size, kernel_size, stride, stride, pad, pad))
                    :add(nn.SpatialBatchNormalization(dual_c2))
                    :add(nn.ReLU())
                )
        dual = nn.Sequential():add(dual):add(nn.JoinTable(2))
        return dual
    end

    local function DualNoR(in_c, dual_c1, dual_c2, kernel_size, stride, pad)
        local dual = nn.ConcatTable()
        dual:add(nn.Sequential()
                    :add(nn.SpatialConvolution(in_c, dual_c1, kernel_size, kernel_size, stride, stride, pad, pad))
                    :add(nn.SpatialBatchNormalization(dual_c1))
                )
        dual:add(nn.Sequential()
                    :add(nn.SpatialConvolution(in_c, dual_c2, kernel_size, kernel_size, stride, stride, pad, pad))
                    :add(nn.SpatialBatchNormalization(dual_c2))
                )
        dual = nn.Sequential():add(dual):add(nn.JoinTable(2))
        return dual
    end

    local input_channels = joint_num+compo_num+3
    local m = nn.Sequential()
--    local dual1 = Dual(input_channels, first_layer_channels, first_layer_channels, 3, 1, 1)
--    local dual2 = Dual(first_layer_channels*2, first_layer_channels*2, first_layer_channels*2, 5, 1, 2)
--    local dual3 = Dual(first_layer_channels*4, first_layer_channels*4, first_layer_channels*4, 5, 1, 2)
    local dual1 = Dual(input_channels, first_layer_channels, first_layer_channels, 3, 1, 1)
    local cbr1 = nn.Sequential()
                    :add(nn.SpatialConvolution(first_layer_channels*2, first_layer_channels*2, 1, 1, 1, 1, 0, 0))
                    :add(nn.SpatialBatchNormalization(first_layer_channels*2))
                    :add(nn.ReLU())
    local dual2 = Dual(first_layer_channels*2, first_layer_channels*2, first_layer_channels*2, 3, 1, 1)
    local cbr2 = nn.Sequential()
                    :add(nn.SpatialConvolution(first_layer_channels*4, first_layer_channels*4, 1, 1, 1, 1, 0, 0))
                    :add(nn.SpatialBatchNormalization(first_layer_channels*4))
                    :add(nn.ReLU())
    local dual3 = Dual(first_layer_channels*4, first_layer_channels*4, first_layer_channels*4, 3, 1, 1)
    local cbr3 = nn.Sequential()
                    :add(nn.SpatialConvolution(first_layer_channels*8, first_layer_channels*8, 1, 1, 1, 1, 0, 0))
                    :add(nn.SpatialBatchNormalization(first_layer_channels*8))
                    :add(nn.ReLU())
    local dual4 = Dual(first_layer_channels*8, first_layer_channels*4, first_layer_channels*4, 3, 1, 1)
    local cbr4 = nn.Sequential()
                        :add(nn.SpatialConvolution(first_layer_channels*8, first_layer_channels*8, 1, 1, 1, 1, 0, 0))
                        :add(nn.SpatialBatchNormalization(first_layer_channels*8))
                        :add(nn.ReLU())
    local dual5 = Dual(first_layer_channels*8, first_layer_channels*4, first_layer_channels*4, 1, 1, 0)
    local cbr5 = nn.Sequential()
                        :add(nn.SpatialConvolution(first_layer_channels*8, first_layer_channels*8, 1, 1, 1, 1, 0, 0))
                        :add(nn.SpatialBatchNormalization(first_layer_channels*8))
                        :add(nn.ReLU())
    local dual6 = DualNoR(first_layer_channels*8, joint_num, compo_num, 1, 1, 0)


    m:add(dual1)--:add(cbr1)
     :add(dual2)--:add(cbr2)
     :add(dual3)--:add(cbr3)
     :add(dual4)--:add(cbr4)
     :add(dual5)--:add(cbr5)
     :add(dual6)

    return m
end

local function tool_regressFullNoConcat(joint_num, compo_num, first_layer_channels)
    local input_channels = 3 + joint_num + compo_num
    local output_channels = joint_num + compo_num
    local m = nn.Sequential()
    m:add(nn.SpatialConvolution(input_channels, first_layer_channels, 3, 3, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(first_layer_channels))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels, first_layer_channels*2, 3, 3, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(first_layer_channels*2))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*2, first_layer_channels*4, 3, 3, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, first_layer_channels*4, 3, 3, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, first_layer_channels*4, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(first_layer_channels*4))
     :add(nn.ReLU())
    m:add(nn.SpatialConvolution(first_layer_channels*4, output_channels, 1, 1, 1, 1))
     :add(nn.SpatialBatchNormalization(output_channels))
--     :add(nn.ReLU())
--    m:add(nn.Sigmoid())

    return m
end

local joint_num = 5
local compo_num = 4
--local regress_net = tool_regress(joint_num+compo_num, 64)
local regress_net = tool_regressFullNoConcat(joint_num, compo_num, 64)
--local regress_net = tool_regressFull(joint_num, compo_num, 32)

print(regress_net)
--regress_net:cuda()
--local inp = torch.CudaTensor(1, 3+joint_num+compo_num, 384, 480):fill(1)
--print(inp:size())
--for i=1, 1000 do
--    local outp = regress_net:forward(inp)
--end
local saveDir = '/home/xiaofei/workspace/toolPose/models'
--local modelConf = {type='toolPoseRegressFull', v='256*320_ftblr_head_noConcat' }
local modelConf = {type='toolPoseRegressFull', v='256*320_ftblr_random_head_noConcat'}
--
local saveID = modelConf.type .. '_v' .. modelConf.v
local initModelPath = paths.concat(saveDir, 'model.' .. saveID .. '.init.t7')
torch.save(paths.concat(saveDir, initModelPath), regress_net)
print('saved model ' .. saveID .. ' to ' .. paths.concat(saveDir, initModelPath))

-- toolPoseRegress v=1: inputsize = [384, 480], input only detMap(down_sample=4)
-- toolPoseRegressFull v=1: inputsize = [384, 480], input frame+detMap(down_sample=1)
-- toolPoseRegressFull v=2: inputsize = [256, 320], input frame+detMap(down_sample=1)
-- toolPoseRegressFull v=4: inputsize = [256, 320], input frame+detMap(round)(down_sample=1), without concat
-- toolPoseRegressFull


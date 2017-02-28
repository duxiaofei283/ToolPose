-- normalize the pos to [0,1]
require 'json'
require 'image'
require 'colormap'
require 'data_utils_new'
torch.setdefaulttensortype('torch.FloatTensor')


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

 ---------------------------------------------------------------------------------------------------
local baseDir, seq_info
local seq_info_tab = {}
local frame_format = 'img_%06d_raw.png'

----- icl dataset
--local IC_testBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/IC'
--local IC_saveDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/tool_label'
--local ic_seq_info1 = {seqName = 'icl_data1', seqDir=paths.concat(IC_testBaseDir, 'icl_data1', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1677 }
--local ic_seq_info2 = {seqName = 'icl_data2', seqDir=paths.concat(IC_testBaseDir, 'icl_data2', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=994 }
--local ic_seq_info3 = {seqName = 'icl_data3', seqDir=paths.concat(IC_testBaseDir, 'icl_data3', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1799}
----seq_info = ic_seq_info3
--seq_info_tab = {ic_seq_info1,ic_seq_info2,ic_seq_info3 }
--local test_anno_tab = genTestData(seq_info_tab)
--print(#test_anno_tab)
--torch.save(paths.concat(IC_saveDir, 'test_icl_frames.t7'), test_anno_tab)
--print('Saved ' .. paths.concat(IC_saveDir, 'test_icl_frames.t7'))

-- -- invivo dataset
--local invivo_testBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/invivo'
--local invivo_saveDir = '/home/xiaofei/public_datasets/MICCAI_tool/Test_data/tool_label'
--local invivo_seq_info1 = {seqName = 'invivo1', seqDir=paths.concat(invivo_testBaseDir, 'invivo1', 'Raw_deinterlace'), frameFormat=frame_format, startFrame=1, endFrame=233 }
--local invivo_seq_info2 = {seqName = 'invivo2', seqDir=paths.concat(invivo_testBaseDir, 'invivo2', 'Raw_deinterlace'), frameFormat=frame_format, startFrame=1, endFrame=659 }
--local invivo_seq_info3 = {seqName = 'invivo3', seqDir=paths.concat(invivo_testBaseDir, 'invivo3', 'Raw_deinterlace'), frameFormat=frame_format, startFrame=1, endFrame=328 }
--local invivo_seq_info4 = {seqName = 'invivo4', seqDir=paths.concat(invivo_testBaseDir, 'invivo4', 'Raw_deinterlace'), frameFormat=frame_format, startFrame=1, endFrame=366 }
--
----seq_info = invivo_seq_info4
--seq_info_tab = {invivo_seq_info1,invivo_seq_info2,invivo_seq_info3,invivo_seq_info4}
--
--local test_anno_tab = genTestData(seq_info_tab)
--print(#test_anno_tab)
--torch.save(paths.concat(invivo_saveDir, 'test_invivo_frames.t7'), test_anno_tab)
--print('Saved ' .. paths.concat(invivo_saveDir, 'test_invivo_frames.t7'))



-- -------------------------------- smoke ----------------
--local cloud_testBaseDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Testing_smoke'
--local cloud_saveDir = cloud_testBaseDir;
--local cloud_seq_info1 = {seqName = 'Dataset1', seqDir=paths.concat(cloud_testBaseDir, 'Dataset1', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=370 }
--local cloud_seq_info2 = {seqName = 'Dataset2', seqDir=paths.concat(cloud_testBaseDir, 'Dataset2', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375 }
--local cloud_seq_info3 = {seqName = 'Dataset3', seqDir=paths.concat(cloud_testBaseDir, 'Dataset3', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375 }
--local cloud_seq_info4 = {seqName = 'Dataset4', seqDir=paths.concat(cloud_testBaseDir, 'Dataset4', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=375 }
--local cloud_seq_info5 = {seqName = 'Dataset5', seqDir=paths.concat(cloud_testBaseDir, 'Dataset5', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1500 }
--local cloud_seq_info6 = {seqName = 'Dataset6', seqDir=paths.concat(cloud_testBaseDir, 'Dataset6', 'Raw'), frameFormat=frame_format, startFrame=1, endFrame=1500 }
--
--seq_info_tab = {cloud_seq_info1, cloud_seq_info2, cloud_seq_info3, cloud_seq_info4, cloud_seq_info5, cloud_seq_info6}
--
--local test_anno_tab = genTestData(seq_info_tab)
--print(#test_anno_tab)
--torch.save(paths.concat(cloud_saveDir, 'test_smoke_frames.t7'), test_anno_tab)
--print('Saved ' .. paths.concat(cloud_saveDir, 'test_smoke_frames.t7'))



-- transfer test label to smoke dataset
local testlabelDir = '/home/xiaofei/public_datasets/MICCAI_tool/Tracking_Robotic_Training/tool_label'
local testlabelFilePath = paths.concat(testlabelDir, 'test_endo_toolpos_head.t7')
local saveDir = testlabelDir

local test_labels = torch.load(testlabelFilePath)
local smoke_labels = {}
for i=1, #test_labels do
    table.insert(smoke_labels, test_labels[i])
    -- change
    local smoke_filename ,change_flag = string.gsub(test_labels[i].filename, '_Testing', '_Testing_smoke')
    assert(change_flag == 1, 'ALERT: filename is not changed.')
    smoke_labels[i].filename = smoke_filename
end
print(#smoke_labels)
torch.save(paths.concat(saveDir, 'test_smoke_toolpos_head.t7'), smoke_labels)
print('Saved ' .. paths.concat(saveDir, 'test_smoke_toolpos_head.t7'))


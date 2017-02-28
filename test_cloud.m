% test adding cloud on image
clear;
close all;
clc;

seq1_frame_range = [1, 370];
seq2_frame_range = [1, 375];
seq3_frame_range = [1, 375];
seq4_frame_range = [1, 375];
seq5_frame_range = [1, 1500];
seq6_frame_range = [1, 1500];

frame_dir = '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Testing/';
seq_type = 'Dataset1/Raw/';

for idx = seq1_frame_range(1):seq1_frame_range(1)
    close all;
    idx
%     save_frame_dir = './';

    img = imread('./frame_raw.png');
%     img = imread(strcat(frame_dir, seq_type, sprintf('img_%06d_raw.png', idx)));
    figure; imshow(img,[]);
    f_name = './original';
    ii = getframe(gcf); imwrite(ii.cdata, [f_name '.png']);

    height = size(img,1);
    width = size(img,2);

    fbm_noise = fbm(width, height);
    smoke_img = get_clouds(fbm_noise);
    % figure; imshow(cloud_img,[]);

    smoke = cat(3, smoke_img,  smoke_img, smoke_img);
    smoke = uint8(smoke);
    figure; imshow(smoke,[]);
    f_name = './smoke';
    ii = getframe(gcf); imwrite(ii.cdata, [f_name '.png']);


    % rgb_cloud = ind2rgb(cloud_img, jet(255));
    % rgb_cloud = 255 * rgb_cloud;
    % figure; imshow(uint8(rgb_cloud),[]);

    fused = double(img) + 0.3 * double(smoke);
    fused = uint8(fused);
    figure; imshow(uint8(fused), []);
    f_name = './fused';
    ii = getframe(gcf); imwrite(ii.cdata, [f_name '.png']);
end
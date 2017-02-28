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
seq_type = 'Dataset6/Raw/';

for idx = seq6_frame_range(1):seq6_frame_range(2)
    close all;
    idx
    save_frame_dir = '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Testing_cloud/';

    img = imread(strcat(frame_dir, seq_type, sprintf('img_%06d_raw.png', idx)));
    %figure; imshow(img,[]);
    % f_name = './original';
    % ii = getframe(gcf); imwrite(ii.cdata, [f_name '.png']);

    height = size(img,1);
    width = size(img,2);

    fbm_noise = fbm(width, height);
    cloud_img = get_clouds(fbm_noise);
    % figure; imshow(cloud_img,[]);

    clouds = cat(3, cloud_img,  cloud_img, cloud_img);
    clouds = uint8(clouds);
    %figure; imshow(clouds,[]);

    % rgb_cloud = ind2rgb(cloud_img, jet(255));
    % rgb_cloud = 255 * rgb_cloud;
    % figure; imshow(uint8(rgb_cloud),[]);

    fused = double(img) + 0.3 * double(clouds);
    fused = uint8(fused);
    figure; imshow(uint8(fused), []);
    f_name = strcat(save_frame_dir, seq_type, sprintf('img_%06d_raw', idx));
    ii = getframe(gcf); imwrite(ii.cdata, [f_name '.png']);
end
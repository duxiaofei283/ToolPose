function a = get_clouds(a)
    % scale a between 0 and 1
    a = a - min(a(:));
    a = a / max(a(:));

    % parameters
    density = 0.4; %0.5;
    sharpness = 0.1; %0.1;

    a = 1-exp((-(a-density)*sharpness));
    a(a<0) = 0;

    % scale between 0 to 255 and quantize

    a = a / max(a(:));
    a = round(a * 255);
   
end

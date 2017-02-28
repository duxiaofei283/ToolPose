function s = fbm(width, height)
% generate FBM noise
    s = zeros(height, width);
    w = width;
    h = height;
    i = 0;
    while w > 3 && h > 3 
        i = i+1;
        d = interp2(rand(h,w), i-1, 'spline');
        s = s + i * d(1:height, 1:width);
        w = w - ceil(w/2-1);
        h = h - ceil(h/2-1);
    end
end
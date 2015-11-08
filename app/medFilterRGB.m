function outImage= medFilterRGB(inImage)

% filter each channel separately
r = medfilt2(inImage(:, :, 1), [3 3]);
%g = medfilt2(inImage(:, :, 2), [3 3]);
%b = medfilt2(inImage(:, :, 3), [3 3]);

% reconstruct the image from r,g,b channels
%outImage = cat(3, r, g, b);
outImage=r;

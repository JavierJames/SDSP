%linear smooth function spatially invariant 
%spatial domain
function outImage = meanFilter(inImage,method)

if method==1
%convert the imgage to double for computation
imgd = im2double(inImage);

%3x3 filter for mean averaging 
filter = ones(3,3)/9;  

%filter the image using mean of surrounding neighbours
imgFiltered = filter2(filter, imgd);
outImage = im2single(imgFiltered);


%method 2
elseif method==2
%convert the imgage to double for computation
imgd = im2double(inImage);

%3x3 filter for mean averaging 
filter = fspecial('average',[3,3]);  

%filter the image using mean of surrounding neighbours
imgFiltered = imfilter(imgd,filter);

outImage = im2single(imgFiltered);
end
%% Script to translate the K-spcae images into spatial eye images
clc; clear all; close all;

%Load the bad data
load('../../MRI_datasets/Slice3/BadData/slice3_channel1.mat');
load('../../MRI_datasets/Slice3/BadData/slice3_channel2.mat');
load('../../MRI_datasets/Slice3/BadData/slice3_channel3.mat');

%Load the good data
%GoodData_ch1 = load('../../MRI_datasets/Slice3/GoodData/slice3_channel1.mat');
load('../../MRI_datasets/Slice3/GoodData/slice3_channel1.mat');
%load('../../MRI_datasets/Slice3/BadData/slice3_channel2.mat');
%load('../../MRI_datasets/Slice3/BadData/slice3_channel3.mat');


% 1. X - dimension of the K-Space data    - 128
% 2. Y - dimension of the K-Space data    - 512


% IFFT of k-space data
%channel 1 (replace "slice1_channel1_goodData" with
%slice1_channel1_badData) for bad images
Data_img(:,:,1) = ifftshift(ifft2(slice3_channel1_badData),1);
%channel 2
Data_img(:,:,2) = ifftshift(ifft2(slice3_channel2_badData),1);
%channel 3
Data_img(:,:,3) = ifftshift(ifft2(slice3_channel3_badData),1);



%plot the histogram of each channel signal
figure; 
subplot(311); imhist(abs(Data_img(:,:,1)))
title('Histogram Bad Image Slice 1');
subplot(312); imhist(abs(Data_img(:,:,2)))
title('Histogram Bad Image Slice 2');
subplot(313); imhist(abs(Data_img(:,:,3)))
title('Histogram Bad Image Slice 3');



test = 0;



%plot how the Forirer and the signal looks in the imageshow and imagesc
%used just for validating and getting understanding 
% figure;
% subplot(221);
% imagesc(100*log(abs(slice3_channel1_badData)));
% subplot(222);
% imshow(slice3_channel1_badData);
% subplot(223);
% imagesc(100*log(abs(Data_img(:,:,1))));
% subplot(224);
% imshow(Data_img(:,:,1));




%plot the energy of the fourier transform and the signal
% figure;
% subplot(121);
% imagesc(100*log(abs(slice3_channel1_badData)));
% subplot(122);
% imagesc(100*log(abs(Data_img(:,:,1))));
%  



 

%----------------------------------------------------------------------
%start
%----------------------------------------------------------------------
% clear compensation, preparation, based on fourier transformed blinked 
% k-space data (Data_raw)
 
eye_visualize  = clearImageSlices(Data_img);


%% plotting scripts
%close all
figure; 
imagesc(eye_visualize(:,:,1));
axis image, 
colormap gray;
axis off

xlabel('Horizontal frequency bins')
ylabel('Vertical frequency bins');


%----------------------------------------------------------------------
%end
%----------------------------------------------------------------------




%----------------------------------------------------------------------
%Method 1: mean filter
%----------------------------------------------------------------------

img1_filtered(:,:,1) = meanFilter(abs(Data_img(:,:,1)),1);  
img1_filtered(:,:,2) = meanFilter(abs(Data_img(:,:,2)),1);  
img1_filtered(:,:,3) = meanFilter(abs(Data_img(:,:,3)),1);  


eye_visualize2  = clearImageSlices(img1_filtered);


%% plotting scripts
%close all
figure; 
imagesc(eye_visualize2(:,:,1));
axis image, 
colormap gray;
axis off

xlabel('Horizontal frequency bins')
ylabel('Vertical frequency bins');





if test ==1

%img1_filtered = meanFilter(Data_img(:,:,1),1);
img1_filtered_slice = meanFilter(abs(Data_img(:,:,1)),1); %actually shows an image
%img1_filtered = meanFilter(Data_img(:,:,1).^2,1); %actually shows an image

% img1_Real = real(Data_img(:,:,1));
% img1_Real_filtered = meanFilter(img1_Real,1); %actually shows an image
% 
% img1_Imag= imag(Data_img(:,:,1));
% img1_Imag_filtered = meanFilter(img1_Imag,1); %actually shows an image
% 
% img1_filtered_2= complex(img1_Real_filtered,img1_Imag_filtered);



%F_img1 = fftshift(fft2(img1_filtered),1);
F_img1 = fft2(fftshift(img1_filtered_slice,1));

 





%compare the original and filtered image 
figure;
subplot(121);
%imagesc(100*log(abs(Data_img(:,:,1))));
%imshow(Data_img(:,:,1));
title('Original image slice');
imagesc(eye_visualize(:,:,1));
axis image, 
colormap gray;
axis off


subplot(122);
%imshow(img1_filtered);
imagesc(100*log(abs(img1_filtered_slice)));
title('Mean filter image slice');
imagesc(eye_visualize(:,:,1));
axis image, 
colormap gray;
axis off

% subplot(133);
% imshow(img1_filtered_2);
% %imagesc(100*log(abs(img1_filtered)));




%compare the fourier transform of the  original and filtered images
figure;
subplot(121);
imagesc(100*log(abs(slice3_channel1_badData)));
%imshow(slice3_channel1_badData);
subplot(122);
imagesc(100*log(abs(F_img1)));
%imshow(F_img1);
 

%calculate the Peak SNR and SNR
GoodImage_c1(:,:,1) = ifftshift(ifft2(slice3_channel1_goodData),1);


psnr_mean_c1 = psnr(Data_img(:,:,1)  , GoodImage_c1 );
psnr_mean_c1_filtered = psnr(img1_filtered_slice  , GoodImage_c1 );
psnr_mean_c1_filtered2 = psnr(img1_filtered_2  , GoodImage_c1 );

psnr_mean_c1_abs = psnr(abs(Data_img(:,:,1))  , abs(GoodImage_c1) );
psnr_mean_c1_filtered_abs = psnr(abs(img1_filtered_slice)  ,abs( GoodImage_c1) );

 
 

%----------------------------------------------------------------------
%end mean filter
%----------------------------------------------------------------------







%----------------------------------------------------------------------
%Method 2: medion filter
%----------------------------------------------------------------------
%split the input image into magnitude and angle
inImageMagnitude= abs(Data_img(:,:,1));
inImageAngle= angle(Data_img(:,:,1));

%process the inputimage Magnitude
%inImage= Data_img(:,:,1);
outImageMagnitude = medFilterRGB(inImageMagnitude);

outImage=outImageMagnitude.*exp(i*inImageAngle);


figure;
subplot(221);
imagesc(100*log(abs(slice3_channel1_badData)));
subplot(222);
imshow(slice3_channel1_badData);
subplot(223);
imagesc(100*log(abs(Data_img(:,:,1))));
subplot(224);
imshow(Data_img(:,:,1));

figure;
subplot(121);
imagesc(100*log(abs(slice3_channel1_badData)));
subplot(122);
imagesc(100*log(abs(Data_img(:,:,1))));
 






% Spatial frequency observations
% figure(2); 
% imagesc(100*log(abs(slice3_channel1_badData)));
% 
% figure(3); 
% imagesc(100*log(abs(slice3_channel2_badData)));
% 
% figure(4); 
% imagesc(100*log(abs(slice3_channel3_badData)));




%inImage is actually the same as the Data_img
inImage=inImageMagnitude.*exp(i*inImageAngle);

inImagefft=fft2(fftshift(inImage,1));
outImagefft=fft2(fftshift(outImage,1));
outImagefft2=fft2(fftshift(outImageMagnitude,1));

Data_imgfft=fft2(fftshift(Data_img(:,:,1),1));

%confirm inImage is the same as Data_img from the 1st channel
figure;
subplot(121);
imagesc(100*log(abs(Data_img(:,:,1))));
title('Data_img');
subplot(122);
imagesc(100*log(abs(inImage)));
title('inImage=abs(Data_img).*exp(angle(Data_img)');

%confirm k-space images
figure;
subplot(131);
imagesc(100*log(abs(slice3_channel1_badData)));
title('kspace channel1');
subplot(132);
imagesc(100*log(abs(Data_imgfft)));
title('kspace Data_img -->fft');
subplot(133);
imagesc(100*log(abs(inImagefft)));
title('kspace inImagefft.');



%confirm space inmage with filtered image 
figure;
subplot(121);
imagesc(100*log(abs(slice3_channel1_badData)));
title('kspace channel1');
subplot(122);
imagesc(100*log(abs(outImagefft)));
title('kspace outImagefft');


%imagesc(100*log(abs(outImagefft)));


%imshow(Data_img(:,:,1));
%figure;
%imshow(inImage);
%figure;
%imshow(outImage);
%subplot(121);imshow(inImage)
%subplot(122);imshow(outImage)

%F = fftshift(ffft2(Data_img(:,:,1),1)); 
%figure; 
%imagesc(100*log(abs(F)));

% figure;
% subplot(221);imshow(Data_img(:,:,1));
% subplot(222);imshow(abs(Data_img(:,:,1)));
% subplot(223);imagesc(100*log(abs(Data_img(:,:,1))));
%subplot(224);imagesc(100*log(Data_img(:,:,1)));



%----------------------------------------------------------------------
%end medion filter
%----------------------------------------------------------------------

end 

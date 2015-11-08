%clear image that consists of 3 slices 
function eye_visualize = clearImageSlices(Image)

% clear compensation, preparation, based on fourier transformed blinked 
% k-space data (Data_raw)
clear_comp = linspace(10,0.1,size(Image,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Image,1) 1]);

% combine 3 channels sum of squares and add clear compensation
eye_raw  = sqrt( abs(squeeze(Image(:,:,1))).^2 + ...
           abs(squeeze(Image(:,:,2))).^2 + ...
           abs(squeeze(Image(:,:,3))).^2).* clear_matrix;  
    
% crop images because we are only interested in eye. Make it square 
% 128 x 128
crop_x = [128 + 60 : 348 - 33]; % crop coordinates
eye_raw = eye_raw(crop_x, :);
 
% Visualize the images. 

% %image
eye_visualize = reshape(squeeze(eye_raw(:,:)),[128 128]); 


% For better visualization and contrast of the eye images, histogram based
% compensation will be done 

std_within = 0.995; 
% set maximum intensity to contain 99.5 % of intensity values per image
[aa, val] = hist(eye_visualize(:),linspace(0,max(...
                                    eye_visualize(:)),1000));
    thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));
    
% set threshold value to 65536
eye_visualize = uint16(eye_visualize * 65536 / thresh); 
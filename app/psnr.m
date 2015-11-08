%calculate the peak signal to noise ration 
function PSNR = psnr(A,ref)


error = ref-A; %pixel error 

%calculate mean square error 
D =  abs(error).^ 2;  % squared absolute error per pixel
MSE= sum(D(:))/numel(ref); % average total squared abs error 


%calculate PSNR
A=255*255./MSE;
PSNR= 10*log10(A);
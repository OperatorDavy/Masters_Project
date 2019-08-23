function [truthImagesOut, resampledImagesOut] = resample_undersample_data_David(truthImagesIn, resampledImagesIn)

sizeIn = nargin;
newMatrix = 128

origMatrix = size(truthImagesIn, 1)

% this bit find the moving heart and uses this as centre for cropping
r_d_fft = itok(truthImagesIn,3);
mid = ceil(size(r_d_fft, 3) / 2);
r_d_fft(:,:,mid-1:mid+1) = 0;             
moving_heart = abs(sum(r_d_fft,3));

m_h_x = sum(moving_heart,1);
m_h_y = sum(moving_heart,2);

com_x = (sum(m_h_x .* [1:length(m_h_x)]))/sum(m_h_x);
com_y = (sum(m_h_y .* [1:length(m_h_y)]'))/sum(m_h_y);

addY = 0;
startY = round(com_y)-(newMatrix/2);
if(startY < 1) 
    addY = -startY + 1;
    startY = 1;
end;
endY = addY + round(com_y)+(newMatrix/2 - 1);
if(endY > origMatrix)
    endY = origMatrix;
    startY = (newMatrix/2 +1);
end

addX = 0;
startX = round(com_x)-(newMatrix/2);
if(startX < 1) 
    addX = -startX + 1;
    startX = 1;
end;
endX = addX + round(com_x)+(newMatrix/2 - 1);
if(endX > origMatrix)
    endX = origMatrix;
    startX = newMatrix/2 +1;
end

% this does some resampling aong time so that all series has 20 frames

nFrames = size(truthImagesIn, 3)
[x,y,z] = meshgrid(1:newMatrix,1:newMatrix,1:nFrames);
[x1,y1,z1] = meshgrid(1:newMatrix,1:newMatrix, 1:(nFrames-1)/19:nFrames);

if(size(x1, 3) ~= 20)
    disp('ERROR');
end


%% JAS start
truthImagesOut1 = double(truthImagesIn(startY:endY,startX:endX,:));
norm_dat = interp3(x,y,z, (truthImagesOut1),x1,y1,z1);

min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));

truthImagesOut = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
truthImagesOut = cast(truthImagesOut, 'single');

%% JAS end

resampledImagesOuta = double(resampledImagesIn(startY:endY,startX:endX,:));
%% JAS start
norm_dat = interp3(x,y,z, (resampledImagesOuta),x1,y1,z1);
min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));

resampledImagesOut = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
resampledImagesOut = cast(resampledImagesOut, 'single');

return;

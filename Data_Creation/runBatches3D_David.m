load 'TrainingData.mat' % also run for 'TestData.mat'

addpath 'GRIDDING/gridder'
%%
addpath 'GRIDDING/gridder'

bIsGA       = true;
bIsTinyGA   = true;
acc_fact    = 20;

 nVolumes = size(new_dat, 2)

 for(b=1:1200)
                
        images = new_dat{1, b};
        disp(['TEST; b = ', int2str(b), ' , nFrames = ', int2str(size(images, 3))]);
 
        disp('tiny golden angle rotating trajectory');
        % this function does the resampling/undersampling onto the radial
        % trajectxory
        [Test13_tGA_rot, gridded_k_data, trajectory, weights] = SimulatingUndersampledRadialData(images, acc_fact, bIsGA, bIsTinyGA, true);
        Test13_tGA_rot = abs(Test13_tGA_rot);
        % this function does some cropping, normalisation of pixel values
        % and resampling along time so that all series have the same number
        % of frames
        
        mask = get_mask(trajectory, 9);
        
        [mask_data(:,:,:,b), imagesResampled__(:,:,:,b)] = resample_undersample_data_David(mask, Test13_tGA_rot); 
        [k_data(:,:,:,b), imagesResampled__(:,:,:,b)] = resample_undersample_data_David(gridded_k_data, Test13_tGA_rot); 
        [imagesTruth(:,:,:,b), imagesResampled(:,:,:,b)] = resample_undersample_data_David(images, Test13_tGA_rot); 
 end
 
 
 
 %saving this file all at once will crash you computer
 %;save('Training_20und_K','k_data', 'mask_data');


% note Andreas network required the data to be in format t, x, y, pt
% so would need to do a permute [3 1 2 4]
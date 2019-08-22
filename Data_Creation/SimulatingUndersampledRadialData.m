function [image_out, gridded_k_data, trajectory, weights] = SimulatingUndersampledRadialData(image_in, acc_fact, bIsGA, bIsTinyGA, bRotateTraj)

%close all;
%addpath 'gridder'

    col_len = size(image_in, 1);
    phs_len = size(image_in, 3);
        
    %----------------------------------------------------------------------
   %  Calculate predicted traj + weights
    %----------------------------------------------------------------------
                                     
    % this fucntion calcualtes the radial trajectory for the new data
    [trajectory, weights] = CalculateRadialTrajectoryDL(col_len, phs_len, acc_fact, bIsGA, bIsTinyGA, bRotateTraj);
    
        %----------------------------------------------------------------------
        % Now sample the k-space data with this trajectory
        %----------------------------------------------------------------------

        k_image_in     = itok(image_in, [1 2]);
        sampled_data   = grid_data_bck(trajectory, k_image_in, [0 0 1]);   
    
%        disp('sampled data calculated');
        %----------------------------------------------------------------------
        % Now grid all data
        %----------------------------------------------------------------------   
      
        gridded_k_data   = grid_data(trajectory, sampled_data, weights, [col_len, col_len, phs_len], [0 0 1]);                                  
%        disp('data regridded');
    
        image_out = ktoi(gridded_k_data, [1, 2]);

        % there is some scaling problem i am not sure about.....
        image_out = image_out / (col_len/(acc_fact*10.0));
    
%        figure;imagesc(abs(image_in(:,:,1)))
%        figure;imagesc(abs(image_out(:,:,1)))


%    end
return
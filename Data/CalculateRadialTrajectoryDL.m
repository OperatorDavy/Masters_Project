function [trajectory, weights] = CalculateRadialTrajectoryDL(col_len, phs_len, acc_fact, bIsGA, bIsTinyGA, bRotateTraj)

%       col_len is matrix size (assumes it is square)
%       nRadialSpokesFS is no of fully sampled radial spokes
%       phs_len is no of phases (time_points)
%       bIsGA is a bool used to determine if it is golden angle data

if(col_len ~= 192)
    disp('check calculation for nRadials');
end

    nRadialSpokesFS     = calculateNoRadials(192, 192, 320);    %=181              % assumes no of spiral required to foill k-space is the same as the matrix size
    nRadialSpokes_ACC   = floor(nRadialSpokesFS / acc_fact);
	
    nRadialSpokesFS = nRadialSpokes_ACC*acc_fact;
    
    npoints     = nRadialSpokes_ACC * col_len * phs_len; 
    dimensions  = 3;
 
    trajectory  = zeros(npoints, dimensions);  % kx, ky, t
	weights     = zeros(npoints, 1);
    
    angle = pi / nRadialSpokesFS;            %regular sampling
    if(bIsGA)
       angle =  (2.0*pi) / (sqrt(5.0) + 1.0);    % ga sampling
       if(bIsTinyGA)
          angle = pi / ((sqrt(5.0)+1.0) / 2.0 + (7.0-1.0));
       end
    end
    
    angle = - angle; % this is now the same as the way the radial spokes rotate on the scanner
    
	for (phs = 1 : phs_len)
        for (lin = 1 : nRadialSpokes_ACC)
			
            if(bRotateTraj)
                % Sampled projection
                if(bIsGA)
                    cur_int = (lin-1) + nRadialSpokes_ACC*(phs-1);           % this continuously counts in the case of GA
                    % note that this does not take account of different slices
                    %       disp('check cur_int as does not rotate for different slices');
                else
                    cur_int = ((lin-1)*acc_fact) + mod((phs-1), acc_fact);   % this rotated the undersampled lines in subsequent frames
                end
            else
                % Sampled projection
                if(bIsGA)
                    cur_int = (lin-1); % + nRadialSpokes_ACC*(phs-1);           % this continuously counts in the case of GA
                    % note that this does not take account of different slices
                    %       disp('check cur_int as does not rotate for different slices');
                else
                    cur_int = ((lin-1)*acc_fact); % + mod((phs-1), acc_fact);   % this rotated the undersampled lines in subsequent frames
                end
            end
            cos_angle = cos(cur_int*angle);
            sin_angle = sin(cur_int*angle);
             
            for (col = 1: col_len)  
                kx = (col-1) - (col_len/2);

                p = (phs-1)*nRadialSpokes_ACC*col_len + (lin-1)*col_len + col; % over_sampling
                
				trajectory(p, 1) = cos_angle*kx;
				trajectory(p, 2) = sin_angle*kx;						
                trajectory(p, 3) = (phs-1) - floor(phs_len/2);
                %disp('CHECK TRAJ PHS VALUE')
                        
				if(kx == 0.0)
					weights(p, 1) = 0.25;
                else
					weights(p, 1) = abs(kx);
                end
            end
        end    
    end
return;
   cols = 20
   p = colormap(hsv(cols)) 

        
   for (phs = 1 : phs_len)
       figure;
       pos = mod(phs,cols);
       if(pos == 0) pos = 1; end
            
       curp=p(pos, :);
       for(i=1:nRadialSpokes_ACC)
           hold on;plot(trajectory((phs-1)*nRadialSpokes_ACC*col_len + (col_len*(i-1))+1: (phs-1)*nRadialSpokes_ACC*col_len + (col_len*i),1), trajectory((phs-1)*nRadialSpokes_ACC*col_len + (col_len*(i-1))+1: (phs-1)*nRadialSpokes_ACC*col_len + (col_len*i), 2), 'color', p(pos, :));
       end
       pause(0.1);
           
%       F(phs) = getframe();
   end
    
%    figure;
%   movie(F)
        
return
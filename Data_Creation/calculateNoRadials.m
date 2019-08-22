% Demo of generating golden angle radial sampling for elliptical FOV support.
% (c) Apr. 2015, Ziyue Wu, University of Southern California


function [nFSRadials] = calculateNoRadials(matrixX, matrixY, rFOV)%, col)
    %clear; 
    %close all;

    opxres = matrixX; % samples per readout 
    X = matrixX; Y = matrixY; % FOV in units of # of pixels

    kmax		= 1.0/(rFOV/matrixX);		%	Maximum k-space extent (/cm)		
    kmax		= kmax/2.0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Part. 3 Golden angle

    %elliptical FOV with isotropic spatial resolution; REGULAR SAMPLING 
    % My own implementation of Larson's method
    n = 1;
    theta_ellip(1) = 0;
    theta_cutoff = pi;
    theta_width =pi;

    while (theta_ellip(n) < pi)
      fov = (cos(theta_ellip(n) + pi/2 )^2/X^2 + sin(theta_ellip(n) + pi/2)^2/Y^2)^(-1/2);    
      dtheta_approx = 1 / ( kmax * fov);
      fov = (cos(theta_ellip(n)+ dtheta_approx/2 + pi/2)^2/X^2 + sin(theta_ellip(n)+ dtheta_approx/2 + pi/2)^2/Y^2)^(-1/2);
      dtheta = 1 / ( kmax * fov);
      theta_ellip(n+1) = theta_ellip(n) + dtheta;
      n = n+1;
    end

    % adjust theta for symmetry
    % choose adjustment based on which spoke is closest to pi
    if (theta_ellip(end) - (theta_cutoff) > (theta_cutoff) - theta_ellip(end-1))
        nFSRadials = n - 2;
%        theta_ellip = theta_ellip(1:end-2)*theta_width/theta_ellip(end-1);
    else
        nFSRadials = n - 1;
%        theta_ellip = theta_ellip(1:end-1)*theta_width/theta_ellip(end);
    end

    
return;
    
kmax2 = kmax * ones(size(theta_ellip));
kmax2 = repmat(kmax2,[opxres 1]);
k =  -1:(2/(opxres-1)):1;
k = repmat(k,[size(kmax2,2) 1]);
k = k.';
k = k .* kmax;

thetaREG = repmat(theta_ellip,[opxres 1]);

hold on;
for i = 1:size(theta_ellip,2)
    p = polar(thetaREG(:,i), k(:,i));
    set(p,'color',col,'linewidth',1)
    hold on;
end

return;





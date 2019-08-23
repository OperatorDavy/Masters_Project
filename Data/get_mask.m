function [mask] = get_mask(trajectory, spokes)
%Gets the mask, 1 for sampled, 0 for not sampled
k = size(trajectory);
time_steps = k(1)/(192*spokes);

mask = zeros(192,192,time_steps);
trajectory(:,1)= trajectory(:,1)+ abs(min(trajectory(:,1)))+1;
trajectory(:,2)= trajectory(:,2)+ abs(min(trajectory(:,2)))+1;
trajectory(:,3)= trajectory(:,3)+ abs(min(trajectory(:,3)))+1;

trajectory = floor(trajectory);

k = size(trajectory);
k = k(1);

for i=1:k
   mask(trajectory(i,1), trajectory(i,2), trajectory(i,3)) = 1;
    
    
end



end
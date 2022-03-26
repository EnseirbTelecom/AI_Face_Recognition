function [omega] = main_comp(x, x_mean, U, l)
%MAIN_COMP Summary of this function goes here
%   Detailed explanation goes here

omega = zeros(l,1);

for loop=1:l
    eigenvector = U(:,loop);
    omega(loop) = (x-x_mean).'*eigenvector;
end

end


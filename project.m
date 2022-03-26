function [out] = project(projector, projected)
%projects the projected vector onto the projector vector and
% returns the projected vector after projection

[h,w] = size(projected);
p = length(projector);

if w~=1 && h~=1
    
    if w==p
        out = zeros(p,h);
        for loop=1:h
            out(:,loop) = projector' * projected(loop,:) * projector;
        end
    elseif h==p
        out = zeros(p,w);
        for loop=1:w
            out(:,loop) = projector' * projected(:,loop) * projector;
        end
    else
        error("Dimensions not matching.");
    end
else
    out = projector' * projected * projector;
end

end
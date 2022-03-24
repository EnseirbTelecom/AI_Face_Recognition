function [out] = project(projector, projected)
%projects the projected vector onto the projector vector and
% returns the projected vector after projection

out = projector' * projected * projector;
end
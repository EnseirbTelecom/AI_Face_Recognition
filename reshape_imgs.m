function [reshaped_img] = reshape_imgs(imgs, lines, cols)
%RESHAPE_IMGS permet de recreer une mosaique d'images en 192x168 a partir
%de leur forme vectorielle


[h,w] = size(imgs);
reshaped_img = zeros(192*lines,168*lines);
vects = zeros(h,lines);
ptrv=1;
ptri=1;

% cette boucle assemble les lignes d'images une à une
for l = 1:lines
    vects(:,:) = imgs(:,ptrv:ptrv+cols-1);
    reshaped_img(ptri:ptri+191,:) = reshape_line(vects,6);
    ptrv = ptrv+cols;
    ptri = ptri+192;
end

end

function [reshaped_line] = reshape_line(vects, cols)

reshaped_line = zeros(192,168*cols);
ptr=1;
% cette boucle cree les lignes d'images en assemblant les colonnes une a
% une
for c = 1:cols
    reshaped_line(:,ptr:ptr+167) = reshape(vects(:,c),192,168);
    ptr=ptr+168;
end

end

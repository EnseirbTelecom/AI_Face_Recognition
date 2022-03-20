function [imgOut, imgOutMeaned] = eigenfaces_builder(img, U, l, x_mean)
%rebuild_MCA extrait les l composantes principales de l'image img et les
%rassemble en une seule image

len = length(img);

imgOut=zeros(len,1);

% projection sur tous les vecteurs propres du facespace
for loop=1:l
    eigenvector = U(:,loop);
    imgOut = imgOut + project(eigenvector, img);
end

imgOutMeaned = imgOut + x_mean;
end


function [out] = project(projector, projected)
%projects the projected vector onto the projector vector and
% returns the projected vector after projection

out = projector' * projected * projector;
end
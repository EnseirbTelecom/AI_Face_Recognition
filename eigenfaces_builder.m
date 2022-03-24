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


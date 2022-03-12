% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))]; % ex: yaleB ' 01 '
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)]; % 将 每个192*168的文件读取成32256 的数字， 然后存储, 总共60个文件
    end
end
% Size of the training set
[P,N] = size(data_trn);
% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I); % 
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1];  % 每类图片有多少个

mean_face_ligne = mean(data_trn,2);
mean_face = reshape(mean_face_ligne,192,168);
Image_mean=mat2gray(mean_face);
imwrite(Image_mean,'meanface.bmp','bmp');
figure,
imagesc(mean_face);
colormap(gray);

% centraliser
X = [];
for i = 1:60
    X(:,i) = (data_trn(:,i) - mean_face_ligne)/sqrt(60); % (p * n ) avec p = 32256, n = 60 = > X
end

R = X * X'; % 32256*32256

R_gram = X' * X; % 60 * 60
% [eigenvector,eigenvalue]=eigs(R,60);
[eigenvector,eigenvalue]=eigs(R_gram,60);
U = X * eigenvector * (eigenvector'*X'*X*eigenvector)^(-0.5); % 特征脸 eigenface
figure,
% eigenfaces
for i = 1:60
    subplot(6,10,i);
    t = reshape(U(:,i),192,168);
    imagesc(t);
    colormap(gray);
end

l = 10; % 低维 % 对6个不同的图像进行低维投射
recons = [];
UL = U(:,1:10);
% for i = 1:60
%     for j = 1:l
%         recons(:,i) = recons(:,i) + U(:,j)'*X(:,i)*U(:,j);
%     end
% end
% % recons = recons + mean_face_ligne;
% % recons_schema = reshape(recons,192,168);
% % figure,
% % subplot(321);
% % imagesc(recons_schema);
% % colormap(gray)
% % 
% % subplot(322)
% % image_fisrt = reshape(data_trn(:,1),192,168);
% % imagesc(image_fisrt);
% % colormap(gray);
% % 



    


%Display the database
% F = zeros(192*Nc,168*max(size_cls_trn));
% for i=1:Nc
%     for j=1:size_cls_trn(i)
%           pos = sum(size_cls_trn(1:i-1))+j;
%           F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
%     end
% end
% figure;
% imagesc(F);
% colormap(gray);
% axis off;
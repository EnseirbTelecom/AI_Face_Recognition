% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training2/';
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

lines = max(size_cls_trn); % no. of lines per subplot
cols  = Nc;                % no. of columns per subplot

% mean_face_ligne = mean(data_trn,2);
% mean_face = reshape(mean_face_ligne,192,168);
% Image_mean=mat2gray(mean_face);
% imwrite(Image_mean,'meanface.bmp','bmp');
% figure,
% imagesc(mean_face);
% colormap(gray);

% centraliser
% X = [];
% for i = 1:60
%     X(:,i) = (data_trn(:,i) - mean_face_ligne)/sqrt(60); % (p * n ) avec p = 32256, n = 60 = > X
% end

% meme resultat, plus simple pour matlab qu'une boucle for
X_mean_emp = 1/N * sum(data_trn,2);
X_centered = data_trn - X_mean_emp;
X = 1/sqrt(N) * X_centered;

% R = X * X'; % 32256*32256 ==> je la commente, mon pc meurt en la
% calculant

R_gram = X' * X; % 60 * 60
% [eigenvector,eigenvalue]=eigs(R,60);
[eigenvector,eigenvalue]=eigs(R_gram,N);
U = X * eigenvector * (eigenvector'*X'*X*eigenvector)^(-0.5); % 特征脸 eigenface

U = real(U);

figure(1)
sgtitle("The eigenvectors of U"); % Peut ne pas fonctionner si Matlab < R2018b
% affichage des eigenfaces
nextBoundary = size_cls_trn(1);  % indice auquel on change de classe
currClass    = 1;                % classe a afficher
offset       = 1;                % decalage dans la classe
for i = 1:N
    if i>nextBoundary
        currClass = currClass+1;
        nextBoundary = nextBoundary + size_cls_trn();
        offset=1;
    end
    subplot(cols,lines,(currClass-1)*lines+offset);
    imagesc(U(:,i));
    colormap(gray);
    offset=offset+1;
end

figure(2)
sgtitle("Reshaped eigenfaces");
nextBoundary = size_cls_trn(1);  % indice auquel on change de classe
currClass    = 1;                % classe a afficher
offset       = 1;                % decalage dans la classe
for i = 1:N
    if i>nextBoundary
        currClass = currClass+1;
        nextBoundary = nextBoundary + size_cls_trn();
        offset=1;
    end
    subplot(cols,lines,(currClass-1)*lines+offset);
    
    img = U(:,i);
    img = reshape(img,192,168);
    imagesc(img);
    colormap(gray);
    offset=offset+1;
end

%% RECONSTRUCTION DES IMAGES

l_values=2:N/6:N; % dimension du facespace, l <= n
Nimg = length(l_values);    % nombre d'images de classe différentes
Nrec = Nc*Nimg;            % Nombre d'images reconstruites au total
imgs  = zeros(P,36);
imgsM = zeros(P,36);

for loop=1:Nrec
    idx = bd(ceil(loop/Nimg));
    [img, imgM]   = eigenfaces_builder(data_trn(:,idx), U, l_values(mod(loop-1,Nimg)+1), X_mean_emp);
    imgs(:,loop)  = img;
    imgsM(:,loop) = imgM;
end

imgs = reshape_imgs(imgs, Nimg,Nimg);
imgsM = reshape_imgs(imgsM,Nimg,Nimg);

% display
f = figure(3);
imagesc(imgs);
% Changement d'axes
ax = get(f,'Children'); % on extrait l'objet Axis de la figure

% intervalles de ticks pour qu'ils soient centres
Xticks = 84:168:1008;
Yticks = 96:192:1152;

% on change les ticks sur les axes X et Y
ax.XTick = Xticks;
ax.YTick = Yticks;

% On extrait les objets NumericRulers qui correspondent aux axes
XA = get(ax,'XAxis');
YA = get(ax,'YAxis');

% On change le texte affiche sur l'axe pour les valeurs souhaitees
XA.TickLabels = l_values;
YA.TickLabels = cls_trn;

colormap(gray);
title("Reconstruction test");
ylabel("Class of the image");
xlabel("Dimension of the facespace");

% display
f = figure(4);
imagesc(imgsM);

% Changement d'axes
ax = get(f,'Children'); % on extrait l'objet Axis de la figure

% on change les ticks sur les axes X et Y
ax.XTick = Xticks;
ax.YTick = Yticks;

% On extrait les objets NumericRulers qui correspondent aux axes
XA = get(ax,'XAxis');
YA = get(ax,'YAxis');

% On change le texte affiche sur l'axe pour les valeurs souhaitees
XA.TickLabels = l_values;
YA.TickLabels = cls_trn;

colormap(gray);
title("Reconstruction test with recentering");
ylabel("Class of the image");
xlabel("Dimension of the facespace");


% img = reshape(img,192,168); % imagesc can't reshape automatically, strange
% fprintf("Generated image with idx=%d and l_value=%d\n",idx,l_values(mod(loop-1,6)+1));
% 
% imagesc(img); colormap(gray);
    



% 
% l = 10; % 低维 % 对6个不同的图像进行低维投射
% recons = [];
% UL = U(:,1:10);



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
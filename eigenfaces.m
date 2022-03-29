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
[cls_trn,bd,~] = unique(lb_trn); %
Nc = length(cls_trn);
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1];  % 每类图片有多少个

lines = max(size_cls_trn); % no. of lines per subplot
cols  = Nc;                % no. of columns per subplot

% meme resultat, plus simple pour matlab qu'une boucle for
X_mean_emp = 1/N * sum(data_trn,2); % mean_face_ligne
X_centered = data_trn - X_mean_emp;
X = 1/sqrt(N) * X_centered; % il n'y a pas le X_mean

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



l_values=2:N/6:N;        % dimension du facespace, l <= n
Nimg = length(l_values); % nombre d'images de classe différentes
Nrec = Nc*Nimg;          % Nombre d'images reconstruites au total
imgs  =  zeros(P,Nrec);
imgsM = zeros(P,Nrec);



for loop=1:Nrec
    idx = bd(ceil(loop/Nimg));
    [img, imgM]   = eigenfaces_builder(data_trn(:,idx), U, l_values(mod(loop-1,Nimg)+1), X_mean_emp);
    imgs(:,loop)  = img;
    imgsM(:,loop) = imgM;
    imgvalue(:,loop) = imgs(:,loop)'*imgs(:,loop);
    imgvalue_original(:,loop) = data_trn(:,idx)'*data_trn(:,idx);
end


imgs = reshape_imgs(imgs, Nimg,Nc);
imgsM = reshape_imgs(imgsM,Nimg,Nc);
ratio = imgvalue./imgvalue_original;
ratio = reshape(ratio,Nimg,Nc).';


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


figure(5)
plot(l_values,ratio);
legende=[];
for loop=1:Nc
    legende = [legende "Classe "+cls_trn(loop)];
end
legend(legende);




%% Classifieur k-NN
adr2 = './database/test1/yaleB09_P00A+010E+00.pgm';
x_images_mat = reshape(double(imread(adr2)),192,168);
x_images = reshape(x_images_mat,32256,1);


% on choisit valeur de l
l = 15;
% on calculer la valeur de wx


% calculer la distance, le prof a dit oublier la forme dans le
% sujet,,, fait ce que il m'expliquer
% au 1er temps, on calculer les distance de tous les images de training
% et on faire sort de l'ordre croissant. donc on peut choisir valeur de
% k, c'est la k valeur premier
[Trainrows,Traincols] = size(X);
[Testrows,Testcols] = size(x_images);



for i = 1:Traincols
    %     for j = 1:k
    wx = Vecteur_composant_principale(l,x_images-X_mean_emp,U);
    wx_train = Vecteur_composant_principale(l,X(:,i),U);
    Vx(i) = sqrt(sum(wx-wx_train).^2);
end
dismin = min(Vx);
Vx_sort = sort(Vx,'ascend');
idx_dismin = find(Vx==dismin);
k = 5; % a choisir

[class, class_decide] = return_class(k,Vx_sort,Vx,lines,cls_trn);

% true_lb = 
% 
% % confmat
% [C,err_rate] = confmat(true_lb,est_lb)


%% pré-classifieur gaussien

l=60;

imgs_cpstes_princ = zeros(l,N);         % vecteurs omega
imgs_cpstes_princ_centr = zeros(l,N);   % vecteurs omega centres
% composantes principales des images
for loop=1:N
    imgs_cpstes_princ(:,loop) = main_comp(data_trn(:,loop), X_mean_emp, U, l);
end

% Moyennes empiriques
card = size_cls_trn(1);
mean = zeros(l,Nc);
interval = 1:card;
for loop = 1:Nc
    card = size_cls_trn(loop);
    mean(:,loop) = 1/card * sum(imgs_cpstes_princ(:,interval),2);
    imgs_cpstes_princ_centr(:,interval) = imgs_cpstes_princ(:,interval) - mean(:,loop);
    interval=interval+card;
end

%covariance empirique
sigma = zeros(60,60);

for i=1:N
    sigma = sigma + imgs_cpstes_princ(:,i) * imgs_cpstes_princ(:,i).';
end

sigma = sigma/N;
sigma_inv = sigma^(-1);

% extraction des individus
individus = [1, 21, 51];
individus_imgs = data_trn(:,individus:individus+card-1);
individus_means_imgs = [mean(:,1) mean(:,3) mean(:,6)];


legendes = ["Image 1","Image 21","Image 51","Moyenne classe 1","Moyenne classe 3","Moyenne classe 6"];

labels=["Projection onto u_1",
    "Projection onto u_2",
    "Projection onto u_3",
    "Projection onto u_4",
    "Projection onto u_5"];

figure(6)

for loop=1:4
    subplot(2,2,loop)
    plot(imgs_cpstes_princ(individus(1),loop),imgs_cpstes_princ(individus(1),loop+1), '* ');
    hold on;
    plot(imgs_cpstes_princ(individus(2),loop),imgs_cpstes_princ(individus(2),loop+1), '* ');
    plot(imgs_cpstes_princ(individus(3),loop),imgs_cpstes_princ(individus(3),loop+1), '* ');
    plot(mean(individus(1),loop),mean(individus(1),loop+1), '+ ');
    plot(mean(individus(2),loop),mean(individus(2),loop+1), '+ ');
    plot(mean(individus(3),loop),mean(individus(3),loop+1), '+ ');
    xlabel(labels(loop));
    ylabel(labels(loop+1));
    legend(legendes);
end

%% CLASSIFIEUR GAUSSIEN SUR BASE D'APPRENTISSAGE

err=0;
err_rate=0;

for idx=1:60
    img_to_classify = data_trn(:,idx); % pour tester
    img_mc = main_comp(img_to_classify, X_mean_emp, U, 60);
    
    pred = zeros(6,1);
    for i=1:Nc % faut trouver pour arreter les boucles for
        pred(i) = (img_mc - mean(:,i)).' * sigma_inv * (img_mc - mean(:,i));
    end
    
    [~,class] = min(pred);
    answer = floor((idx-1)/10)+1;
    fprintf("L'image %d est dans la classe %d\n",idx,class);
    if class~=answer
        err=err+1;
        fprintf("ERREUR! La classe attendue etait %d\n",answer);
    end
end


err_rate = err/N * 100;
fprintf("Resultat sur les donnees d'apprentissage : %d erreurs (%.2f%%)\n",err,err_rate);

%% CLASSIFIEUR GAUSSIEN SUR BASE DE TEST

% test sets
adr1 = './database/test1/';
adr2 = './database/test2/';
adr3 = './database/test3/';
adr4 = './database/test4/';
adr5 = './database/test5/';
adr6 = './database/test6/';

trn_adr = [adr1 ; adr2 ; adr3 ; adr4 ; adr5 ; adr6];

%TST=4; % which test database to use
for TST=1:6
    adr = trn_adr(TST,:);
    fld = dir(adr);
    nb_elt = length(fld);
    % Data matrix containing the training images in its columns
    data_tst = [];
    % Vector containing the class of each training image
    lb_tst = [];
    for i=1:nb_elt
        if fld(i).isdir == false
            lb_tst = [lb_tst ; str2num(fld(i).name(6:7))]; % ex: yaleB ' 01 '
            img = double(imread([adr fld(i).name]));
            data_tst = [data_tst img(:)]; % 将 每个192*168的文件读取成32256 的数字， 然后存储, 总共60个文件
        end
    end
    
    [P,nb_img] = size(data_tst);
    
    err=0;
    err_rate=0;
    for idx=1:nb_img
        img_to_classify = data_tst(:,idx); % pour tester
        img_mc = main_comp(img_to_classify, X_mean_emp, U, 60);
        
        pred = zeros(6,1);
        for i=1:Nc % faut trouver pour arreter les boucles for
            pred(i) = (img_mc - mean(:,i)).' * sigma_inv * (img_mc - mean(:,i));
        end
        
        [~,class] = min(pred);
        answer = find(cls_trn==lb_tst(idx));
        %fprintf("L'image %d est dans la classe %d\n",idx,class);
        if class~=answer
            err=err+1;
            fprintf("ERREUR! La classe attendue etait %d\n",answer);
        end
    end
    
    err_rate = err/nb_img * 100;
    fprintf("Resultat sur les donnees d'entrainement no.%d : %d erreurs (%.2f%%)\n",...
        TST, err,err_rate);
end

 function [C,err_rate] = confmat(true_lb,est_lb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARGUMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% true_lb : vector of n true labels (integers) attached to the data
% est_lb : vector of n estimated labels. The different labels in est_lb 
% must be elements of true_lb.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C : confusion matrix
% err_rate : total error rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = unique(true_lb);
m = length(ind); % number of classes
C = accumarray([true_lb,est_lb],ones(length(true_lb),1));
% length(true_lb) -> 有多少线

% confusion matrixa 
% 笼统的说，是用subs向量中的信息从val中提取数值做累加，累加完的结果放到A中。
% subs提供的信息由两个：
% 
% (a). subs向量中的每个位置对应val的每个位置；
% 
% (b). subs中元素值相同的，val中的对应元素累加，元素值是累加完后放到A的什么地方。subs = [1; 2; 4; 2; 4]  
C = C(ind,ind);
err_rate = sum(sum(C-diag(diag(C))))/sum(sum(C));
C = diag(1./sum(C,2))*C;
end


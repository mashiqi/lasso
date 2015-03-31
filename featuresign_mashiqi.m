function beta = featuresign_mashiqi(X, y, lambda, standardize)
%{
% Feature-Sign algorithm.
% Author: Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
% Date: 1/10/2015
% Version: 2.0
% 
% This code solves the following problem:
%            argmin_(beta) 0.5*||y - X*beta||_2 + lambda*||beta||_1
% 
% Parameter instruction:
% input:
% X: samples of predictors. Each column of X is a predictor, and each row
% is a data sample.
% y: the response.
% lambda: the coefficient of the norm-one term.
% standardize: the indicator. If standardize == 1, every column in X and y
% will be standardized to mean zero and standard deviation 1. And if its
% value is 0, then standardization process will not be executed.
% standardize == 0 as default.
%
% output:
% beta: weight vector.
%
% reference: 
% [1] Lee, Honglak, et al. "Efficient sparse coding algorithms." Advances
      in neural information processing systems. 2006. 
%}

if nargin < 4
    standardize = 0;
end
if standardize == 1
    n = size(X,1); % number of samples
    X = bsxfun(@minus,X,mean(X,1));
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    y = bsxfun(@minus,y,mean(y,1));
end

if lambda == 0
    lambda = 1e-17; % 梯度可能永远都不能达到绝对0，所以放宽一点
end

p = size(X,2); % number of predictors
activeSet = false(p,1);
beta = zeros(p,1);
theta = zeros(p,1);
XTX = X'*X;
optimality_a = false;
optimality_b = false;

% Step 1 :corresponding to the "Algorithm 1" in "Efficient sparse coding
% algorithm"
grad = XTX*beta-X'*y; %梯度向量

% Step 2 :同上
    [~,currentIndex] = max( abs(grad).*(~activeSet) );
    cnt = 2;
while ~optimality_b
    cnt = cnt + 1;
    if grad(currentIndex) > lambda
        theta(currentIndex) = -1;
        activeSet(currentIndex) = true;
    elseif grad(currentIndex) < -lambda
        theta(currentIndex) = 1;
        activeSet(currentIndex) = true;
    else
        return; % 说明已经达到最小值
    end

    % Step 3 :同上
    while ~optimality_a
        betaHat = zeros(p,1);
        betaUpdate = zeros(p,1);
        betaHat(activeSet) = beta(activeSet);
        betaUpdate(activeSet) = XTX(activeSet,activeSet) \ ( X(:,activeSet)'*y - lambda*theta(activeSet) );
        
        % Step 4 :同上
        temp1 = betaUpdate ./ betaHat; % temp1的作用为：判断beta(activeSet)更新前后有哪些分量改变了正负号
        temp1(isnan(temp1))=0; % 将向量temp1中等于NaN的元素置为0值
        temp1(isinf(temp1))=0; % 将向量temp1中等于正负Inf的元素置为0值
        [scale,j] = min( temp1 );
        if scale >= 0 % 说明没有active的系数在这次更新过程中改变正负号
            beta(activeSet) = betaUpdate(activeSet); % 将所求得的局部最优解返给输出beta
            grad = XTX*beta-X'*y; % 更新梯度向量
            [grad_value_abs,currentIndex] = max( abs(grad).*(~activeSet) );
            optimality_a = true;
            if ~isempty(grad_value_abs) && (grad_value_abs <= lambda) % abs(grad_value) <= lambda
                                                               % 当act_idx0为空集时，意味着beta的所有分量现在都不为零
                                                               % 且使grad =0,所以此时的beta就是最优解了，应该跳出循环了。
                optimality_b = true;
            else
                optimality_a = false;
                optimality_b = false;
                break; % 跳出optimality_a循环，重新选择新的零分量，从step2再开始
            end
        else
            % 下面在s与s_new的连线上找最先改变符号的分量的下标
            betaHat = betaHat + ( betaUpdate - betaHat )/(1-scale);
            betaHat(j) = 0; % 确保第一个跨过零点的分量一定要为零
            beta = betaHat; % 将所求得的局部最优解返给输出beta
            theta(j) = 0; % 将此过零点的分量的符号置零
            activeSet(j) = false; % 从active set中清除此分量
            optimality_a = false;
        end
    end % optimality_a
end % optimality_b
return;
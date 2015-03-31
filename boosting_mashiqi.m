function [beta,beta_trace] = boosting_mashiqi(X,y,standardize)
%{
% Boosting algorithm.
% Author: Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
% Date: 1/16/2015
% 
% This function tries to find the proper solution of the following question:
%            argmin_(beta) ||y - X*beta||_2
% 
% Parameter instruction:
% input:
% X: samples of predictors. Each column of X is a predictor, and each row
% is a data sample.
% y: the response.
% standardize: the indicator. If standardize == 1, every column in X and y
% will be standardized to mean zero and standard deviation 1. And if its
% value is 0, then standardization process will not be executed.
% standardize == 0 as default.
%
% output:
% beta: weight vector.
% beta_trace = trace of weight vector.
%
% reference: 
% [1]Efron, Bradley, et al. "Least angle regression." The Annals of 
%    statistics 32.2 (2004): 407-499.
%}

%% parameter check
if nargin < 3
    standardize = 0;
end
if isvector(y) && (size(X,1) ~= size(y,1))
    disp('Y is not a vector, or the length of Y is not equal to the number of row of X');
    return;
end

%% initialization
if standardize == 1
    n = size(X,1); % number of samples
    X = bsxfun(@minus,X,mean(X,1));
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    y = bsxfun(@minus,y,mean(y,1));
end
n = size(X,1); % number of samples
p = size(X,2); % number of predictors
if size(y,1) ~= n
    disp('input do not satisfy!');
    beta = [];
    return;
end
beta = zeros(p,1);
beta_trace  = zeros(p,1);
corSignn = zeros(p,1);
prediction = zeros(n,1); % "prediction" is the current step-forward vector
fullSet = 1:p;
activeSet = [];
residual = y;
XTX = X'*X;
flag = 0;
epss = 1e-3;
epsilon =  norm(y)*1e-3;
CC = [];

%% begin
cor = residual'*X;
[C,index] = max(abs(cor));
CC = [CC,C];
if C < epss % the proximal solution has been  found, it almost is zero.
    beta = zeros(p,1);
    beta_trace = [beta_trace, beta];
    return;
end
cnt = 0;
while (cnt < 50000)
    beta(index) = beta(index) + epsilon*sign(cor(index));
    beta_trace = [beta_trace, beta];
    residual = residual - X(:,index)*epsilon*sign(cor(index));
    cor = residual'*X;
    cnt = cnt + 1;
    if mod(cnt,100) == 0
        figure(2);plot(repmat([1:size(beta_trace,2)]',1,size(beta_trace,1)),beta_trace');
        pause(0.01);
    end
    [C,index] = max(abs(cor));
    CC = [CC,C];
    if C < epss % the proximal solution has been  found, it almost is zero.
        return;
    end
end
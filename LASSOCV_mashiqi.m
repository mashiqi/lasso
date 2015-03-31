function Lambda = LASSOCV_mashiqi(X,y,K,standardize)
%{
% LARS-LASSO Cross-Validation algorithm.
% Author: Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
% Date: 1/18/2015
% Version: 1.0
%
% Parameter instruction:
% input:
% X: samples of predictors. Each column of X is a predictor, and each row
% is a data sample.
% y: the response. y shold be a vertical vector.
% K: K-fold.
% standardize: the indicator. If standardize == 1, every column in X and y
% will be standardized to mean zero and standard deviation 1. And if its
% value is 0, then standardization process will not be executed.
% standardize = 0 as default.
%
% output:
% Lambda: 
%
% reference: 
% http://www.stat.cmu.edu/~ryantibs/datamining/lectures/18-val1-marked.pdf 
%}

%% parameter check
if nargin == 4; % parameter is complete.
    1;
end
if nargin < 4 || isempty(standardize)
    standardize = 0;
end
if nargin < 3
    K = 5;
end
if K <= 0
    LeaveOneOut = 1;
    K = size(X,1);
else
    LeaveOneOut = 0;
end
if nargin < 2
    disp('Parameter invalid, please check it.');
    return;
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
fullIndex = 1:n;
N = 101; % 'N' is the dicrete level
% beta = zeros(p,1);
XFoldsIndex = cell(K,1);
yFoldsIndex = cell(K,1);
[~,history] = LASSO_mashiqi(X,y,[],[],standardize);
lambdaMax = history.Lambda(1);
lambdaSet = linspace(0,lambdaMax,N);

%% seperate data into K folds
if ~LeaveOneOut
    numOfRow = ceil(n/K);
    leaveRow = numOfRow*K-n;
    Index = 1:n;
    t = randsample(Index,(K-leaveRow)*numOfRow);
    XFoldsIndex(1:(end-leaveRow)) = mat2cell(reshape(t,K-leaveRow,numOfRow),ones(K-leaveRow,1),numOfRow);
    yFoldsIndex(1:(end-leaveRow)) = XFoldsIndex(1:(end-leaveRow));
    t = mysetdiff(Index,t);
    numOfRow = numOfRow - 1;
    XFoldsIndex((end-leaveRow+1):end) = mat2cell(reshape(t,leaveRow,numOfRow),ones(leaveRow,1),numOfRow);
    yFoldsIndex((end-leaveRow+1):end) = XFoldsIndex((end-leaveRow+1):end);
else
    XFoldsIndex(1:K) = num2cell(1:K);
    yFoldsIndex(1:K) = num2cell(1:K);
end

%% cross-validation for lasso begins
err = zeros(N,K);
for fold = 1:K
    % get training data
    tempX = X(mysetdiff(fullIndex,XFoldsIndex{fold}),:);
    tempy = y(mysetdiff(fullIndex,yFoldsIndex{fold}),:);
    % training
    [beta,~] = LASSO_mashiqi(tempX,tempy,[],lambdaSet,standardize);
    tempX = X(XFoldsIndex{fold},:);
    tempy = y(yFoldsIndex{fold},:);
    err(:,fold) = sum((repmat(tempy,1,N) - tempX*beta.Lambda).^2,1)'/length(tempy);
end
CV = sum(err,2)/K; CV = CV';
SE = std(err,0,2);
[~,indexOfMin] = min(CV);
Lambda.usualRule = lambdaSet(indexOfMin);
indexOfMin = find( CV < CV(indexOfMin)+SE(indexOfMin) );
indexOfMin = max(indexOfMin); % increasing regularization
Lambda.oneStandardErrorRule = lambdaSet(indexOfMin);
Lambda.lambdaSet = lambdaSet;
Lambda.CV = CV;
Lambda.SE = SE;
errorbar(lambdaSet,CV,SE);

%-- for debug
% [B,STATS] = lasso(X,Y,'Lambda',lambdaSet,'CV',10);
% tempSE = STATS.SE;
% tempLambdaMinMSE = STATS.LambdaMinMSE;
% temp.Lambda1SE = STATS.Lambda1SE;
% 1;
%-- for debug

function C = mysetdiff(A,B)
% MYSETDIFF Set difference of two sets of positive integers (much faster than built-in setdiff)
% C = mysetdiff(A,B)
% C = A \ B = { things in A that are not in B }
%
% Original by Kevin Murphy, modified by Leon Peshkin

if isempty(A)
    C = [];
    return;
elseif isempty(B)
    C = A;
    return; 
else % both non-empty
    bits = zeros(1, max(max(A), max(B)));
    bits(A) = 1;
    bits(B) = 0;
    C = A(logical(bits(A)));
end
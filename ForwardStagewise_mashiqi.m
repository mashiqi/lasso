function [beta,beta_trace] = ForwardStagewise_mashiqi(X,y,standardize)
%{
% LARS-Forward Stagewise algorithm.
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
epss = 1e-4;
epsilon = 5e-4;

%% At the beginning, find the first most current correlate predictor.
cor = zeros(1,p);
cor(mysetdiff(fullSet,activeSet)) = residual'*X(:,mysetdiff(fullSet,activeSet)); % compute the correlations

C = max(abs(cor));
index = find( abs(abs(cor) - C) < epss); % find the current most correlating index
if C < epss % the proximal solution has been  found, it almost is zero.
    beta = zeros(p,1);
    return;
end
corSignn(index) = sign(cor(index))';
activeSet = [activeSet,index]; % updating active set
nonActiveSet = mysetdiff(fullSet,activeSet);

if isempty(nonActiveSet)
    %{
    % if 'nonActiveSet' is empty at this time, it seems that y are
    % equal-correlated with each column of X, namely y already in the
    % equiangular direction at the beginning. Therefore the following is
    % easy.
    %}
    correspondingBeta = XTX\sign(cor)'; % inv(X'*X)*sign(cor)'
    A = 1 / norm(sign(cor)*correspondingBeta); % 1 / norm(sign(cor)*inv(X'*X)*sign(cor)')
    correspondingBeta = A*correspondingBeta;
    gamma = (X(:,1)'*y) / (X(:,1)'*X*correspondingBeta);
    beta = gamma*correspondingBeta;
    beta_trace = [beta_trace, beta];
    return;
end

% step forward
randIndex = randsrc(1,1,activeSet); % randomly choose one index from the active set
beta(randIndex) = beta(randIndex) + epsilon*corSignn(randIndex);
residual = residual - epsilon*X(:,randIndex)*corSignn(randIndex);
cor = residual'*X;

%% main loop
cnt = 0;
while 1 % for repeatrepeat = 1:min(n,p)
    % check whether actives set needs to be update
    c = max(abs(cor(nonActiveSet)));
    j = ( abs(cor(nonActiveSet)) == c );
    if c > min(abs(cor(activeSet)))
        j = nonActiveSet(j);
        activeSet = [activeSet, j];
        nonActiveSet = mysetdiff(nonActiveSet, j);
        corSignn(j) = sign(cor(j));
        activSetchanged = 1;
    else
        activSetchanged = 0;
    end
    if activSetchanged == 1
        % recompute the equiangular vector and correponding beta
        signActive = corSignn(activeSet);
        correspondingBeta = zeros(p,1);
        correspondingBeta(activeSet) = XTX(activeSet,activeSet)\signActive; % inv(XTX(activeSet,activeSet))*signActive
        A = 1 / sqrt(signActive'*correspondingBeta(activeSet));
        correspondingBeta = A*correspondingBeta;
        equiAngularVec = X(:,activeSet)*correspondingBeta(activeSet);
        %{
        % 'correspondingBeta' and 'equiAngularVec'are only useful to find
        % out if there are indexes that violate (6.3) of the reference
        % paper
        %}
        % find out the index that violate (6.3) of the reference paper
        signEqual = zeros(p,1);
        signEqual(activeSet) = corSignn(activeSet) - sign(correspondingBeta(activeSet));
        signNotEqualIdx = find(signEqual ~= 0)';
        if ~isempty(signNotEqualIdx)
            % there are violate index(es), then updates some variables
            activeSet = mysetdiff(activeSet, signNotEqualIdx);
            nonActiveSet = [nonActiveSet, signNotEqualIdx];
            corSignn(signNotEqualIdx) = 0;
        end  
        if isempty(nonActiveSet)
            % now all beta_i will not change their sign, and they are just
            % remains to develop non-negatively to their final solution.
            flag = 1;
        end
    end
    
    % step forward
    C = max(abs(cor(activeSet)));
    maxSet = (abs(cor(activeSet)) == C); maxSet = activeSet(maxSet);
    randIndex = randsrc(1,1,maxSet); % randomly choose one index from the active set
    beta(randIndex) = beta(randIndex) + epsilon*corSignn(randIndex);
    beta_trace = [beta_trace, beta];
    residual = residual - epsilon*X(:,randIndex)*corSignn(randIndex);
    cor = residual'*X;
    cnt = cnt + 1;
    if mod(cnt,500) == 0
        figure(2);plot(repmat([1:size(beta_trace,2)]',1,size(beta_trace,1)),beta_trace');
    end
%     if flag == 1
%         % the final forward step
%         deltaBeta = (XTX)\X'*residual;
%         beta = beta + deltaBeta;
%         beta_trace = [beta_trace, beta];
%         residual = residual - X*deltaBeta;
%         cor = residual'*X;
%         break;
%     end
    if  norm(cor) < epss
        break;
    end
end

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
function [beta,beta_trace] = LARS_mashiqi(X,y,standardize)
%{
% Least angle regression (LAR) algorithm.
% Author: Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
% Date: 1/8/2015
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
signn = zeros(p,1);
prediction = zeros(n,1); % "prediction" is the current step-forward vector
fullSet = 1:p;
activeSet = [];
residual = y;
XTX = X'*X;
epss = 1e-10;

%% At the beginning, find the first most current correlate predictor.
cor = zeros(1,p);
cor(mysetdiff(fullSet,activeSet)) = residual'*X(:,mysetdiff(fullSet,activeSet)); % compute the correlations

C = max(abs(cor));
index = find( abs(abs(cor) - C) < epss); % find the current most correlating index
if C < epss % the proximal solution has been  found, it almost is zero.
    beta = zeros(p,1);
    return;
end
signn(index) = sign(cor(index))';
activeSet = [activeSet,index]; % updating active set
nonActiveSet = mysetdiff(fullSet,activeSet);

if isempty(nonActiveSet)
    % if 'nonActiveSet' is empty at this time, it seems that y are
    % equal-correlated with each column of X, namely y already in the
    % equiangular direction at the beginning. Therefore the following is
    % easy.
    correspondingBeta = XTX\sign(cor)'; % inv(X'*X)*sign(cor)'
    A = 1 / norm(sign(cor)*correspondingBeta); % 1 / norm(sign(cor)*inv(X'*X)*sign(cor)')
    correspondingBeta = A*correspondingBeta;
    gamma = (X(:,1)'*y) / (X(:,1)'*X*correspondingBeta);
    beta = gamma*correspondingBeta;
    beta_trace = [beta_trace, beta];
    return;
end

%% main loop
while 1 % for repeatrepeat = 1:min(n,p)
    % compute the equiangular vector
    signActive = signn(activeSet);
    correspondingBeta = zeros(p,1);
    correspondingBeta(activeSet) = XTX(activeSet,activeSet)\signActive; % inv(XTX(activeSet,activeSet))*signActive
    A = 1 / sqrt(signActive'*correspondingBeta(activeSet));
    correspondingBeta = A*correspondingBeta;
    equiAngularVec = X(:,activeSet)*correspondingBeta(activeSet);
    %{
    In order to find out the solution beta, we should map this
    'equiAngularVec' to its corresponding 'beta':
    X * correspondingBeta = equiAngularVec
    %}
    
    % To find out the next active predictor.
    cor = zeros(1,p);
    cor(nonActiveSet) = residual'*X(:,nonActiveSet); % compute the correlations between predictors and residual
    C = abs(residual'*X(:,activeSet(1))); % compute the biggest correlations
    if C < epss % the proximal solution has been found. Now it's time to return.
        return;
    end
    cor_equi = zeros(1,p);
    cor_equi(nonActiveSet) = equiAngularVec'*X(:,nonActiveSet); % compute the correlations between predictors and the equiangular vector
    gammaTemp = Inf(2,p);
    gammaTemp(:,nonActiveSet) = [C - cor(nonActiveSet); C + cor(nonActiveSet)] ./ [A - cor_equi(nonActiveSet); A + cor_equi(nonActiveSet)];
    gammaTemp(gammaTemp<0) = Inf;
    
    gammaMin = min(min(gammaTemp(:,nonActiveSet))); % there may be more than one result. 比如说，刚好处在某两个向量的角平分线上，那么这两个的gamma值就是相等的
    if max(abs((residual-gammaMin*equiAngularVec)'*X(:,activeSet(1)))) < epss % 如果这里的if语句能通过的话，说明residual已经与当前的prediction正交，所以应该要return了。如果不return，后面就会因为计算精度问题而导致错误结果
        % Now after the next step 'gammaMin*equiAngularVec' is executed,
        % all the predictor will almost orthogonal to the residual. It
        % means that the proximal beta has been found, so we should return
        % there. If we do not finish this function there, some wrong
        % results will happen.
        beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
        prediction = X(:,activeSet)*beta(activeSet); % compute the current prediction vector
        residual = y - prediction; % compute the current residual
        nonActiveSet = mysetdiff(fullSet,activeSet);
        beta_trace = [beta_trace,beta];
        return;
    end
    index1 = find( abs(gammaTemp(1,:)-gammaMin) < epss ); % find(gammaTemp(1,:) == gammaMin);
    signn(index1) = 1; %-- add code
    index2 = find( abs(gammaTemp(2,:)-gammaMin) < epss ); % find(gammaTemp(2,:) == gammaMin);
    signn(index2) = -1; %-- add code
%     signn(index2) = -signn(index2);
    index = [index1, index2]; % 向前走的同时又能找到下一个(或多个)predictor，这两个步骤是同时完成的
    
    % Now the next active predictor has been found out, then we should make
    % prediction.
    beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
    prediction = X(:,activeSet)*beta(activeSet); % compute the current prediction vector
    residual = y - prediction; % compute the current residual
    activeSet = [activeSet,index]; % now the new predictor is added to the active set
    nonActiveSet = mysetdiff(fullSet,activeSet);
    beta_trace = [beta_trace,beta];
    
    if isempty(nonActiveSet)
        % Now all of these predictors have been active, the prediction
        % vector should get to its final destination to finish its duty.
        % compute the equiangular vector
        signActive = signn(activeSet);
        correspondingBeta(activeSet) = XTX(activeSet,activeSet)\signActive; % inv(XTX(activeSet,activeSet))*signActive
        A = 1 / sqrt(signActive'*correspondingBeta(activeSet));
        correspondingBeta = A*correspondingBeta;
        equiAngularVec = X(:,activeSet)*correspondingBeta(activeSet);
        gammaMin = residual'*equiAngularVec; % this is the projection length
        beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
        prediction = X(:,activeSet)*beta(activeSet); % compute the current prediction vector
        residual = y - prediction; % compute the current residual
        beta_trace = [beta_trace, beta];
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
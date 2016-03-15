function [beta,history] = LASSO_mashiqi(X,y,T,Lambda,standardize)
%{
% LARS-LASSO algorithm.
% 
% 
%	Author:			- Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
%	Date:			- 3/15/2016
%	Version:		- 2.5
% 
% 
% This function tries to find the proper solution of the following problem:
% problem(1): argmin_(beta) ||y - X*beta||^2 s.t. ||beta||_1 <= T
% or:
% problem(2): argmin_(beta) 0.5*||y - X*beta||^2 + Lambda*||beta||_1
% 
% 
% INPUT ARGUMENTS:
% 
%	X				- samples of predictors. Each column of X is a predictor, and each row
% 					  is a data sample.
% 
% 	y				- the response. y shold be a vertical vector.
%
% 	T				- the norm-one type (1)constraint. T can be a vector. T = [] as default.
%
% 	Lambda			- the corresponding type (2) constraint. Lambda can be a vector.
% 					  Lambda = [] as default.
%
% 	standardize		- the indicator. If standardize == 1, every column in X and y
% 					  will be standardized to mean zero and standard deviation 1. And if its
% 					  value is 0, then standardization process will not be executed.
% 					  standardize = 0 as default.
%
%
% OUTPUT ARGUMENTS:
%
% 	beta			- results:
%
%						[1] beta.TLambdaPairs: the first row are values of Ts, and the second row
%							are the values of Lambdas. T and Lambda which are in the same column 
%							have same solution respectively to 'problem(1)' and 'problem(2)'
%
%						[2] beta.T: each column is the corresponding solution to 'T' specified in
%							the input argument
%
%						[3] beta.Lambda: each column is the corresponding solution to 'Lambda' 
%							specified in the input argument
%
% 	history			- the history of solution path. 'history' records the status in  
% 					  every breakpoint by recording the following value:
%
%   					[1] history.beta_trace: every beta value in the breakpoint, including
%   						the starting zero point.
%
%   					[2] history.correspondingBeta: every corresponding beta of the
%   						equiangular vector. The 'correspondingBeta' also represents the \Delta
%   						beta: 
%           					beta_(next breakpoint) = beta_(crrent breakpoint) +
%           											 constant*correspondingBeta.
%
%   					[3] history.activeSet: stores the active sets. Every active set in the
%   						breakpoints will be stored in this cell-type variable. When you want to
%   						access the kth active set, you should input "history.activeSet{k+1}".
%
%   					[4] history.T: store the one-norm of beta in every breakpoint,
%   						including the starting zero point.
%
%   					[5] history.Lambda: store the corresponding Lambda.
%
%   					[6] history.A: if you read the reference paper below, you will know
%   						what is value is. It's difficult to explain if you didn't read this
%   						paper.
%
%   					[7] history.stopReason: a string type. Indicate in what situation is
%   						the function returned.
% 
% 
% EXAMPLE:
% 
% This function computes every beta value in the breakpoint and several
% related values. If you just want to get the specific beta for a specific
% constraint T, then this function maybe too cast in time spend. You can
% run the following code to get started:
%            [beta,history] = LASSO_mashiqi(X,y);
% where the 'beta' gives the Least Square solution, or
%            [beta,history] = LASSO_mashiqi(X,y,T);
% where the variable T is the norm-one constraint and 'beta.T' gives the
% lasso solution under this norm-one constraint. If you want to
% pre-standardize the input data X and y, then you'd better use:
%            [beta,history] = LASSO_mashiqi(X,y,T,[],1); or
%            [beta,history] = LASSO_mashiqi(X,y,[],[],1);
% then you will get the results.
% There is a simple example:
%            clear;
%            X = randn(200,50);
%            beta = randn(50,1);
%            y = X*beta;
%            T = Inf;
%            standardize = 0;
%            [beta_hat1,history] = LASSO_mashiqi(X,y,T,[],standardize);
%            difference1 = norm(beta-beta_hat1);
%            disp(difference1);
%            T = norm(beta,1) / 2; % an active norm-one constraint
%            [beta_hat2,history] = LASSO_mashiqi(X,y,T,[],standardize);
%            difference2 = abs(norm(beta_hat2,1) - T);
%            disp(difference2);
%            T = [0.1 0.2 0.3];
%            Lambda = [0.2 0.3 0.4];
%            [beta,history] = LASSO_mashiqi(X,y,T,Lambda);
%            beta_hat2.T
%            beta_hat2.Lambda
% That's it.
% 
% 
% REFERENCE:
% 
% [1] Efron, Bradley, et al. "Least angle regression." The Annals of 
%    statistics 32.2 (2004): 407-499.
%}

%% parameter check
if nargin == 5; % parameter is complete.
    1;
end
if nargin < 5 || isempty(standardize)
    standardize = 0;
end
if nargin < 4
    Lambda = [];
end
if nargin < 3
    T = [];
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
beta = zeros(p,1);
fullSet = 1:p;
activeSet = [];
signn = ones(p,1);
residual = y;
LASSOWorks = 0;
XTX = X'*X;
prediction = zeros(n,1); % "prediction" is the current step-forward vector
epss = 1e-12;
history.beta_trace = zeros(p,1);
history.correspondingBeta = [];
history.activeSet{1} = []; % cell class
history.T = 0;
history.Lambda = 0;
history.A = [];
history.stopReason = [];

%% At the beginning, find the first most current correlate predictor.
cor = zeros(1,p);
cor(mysetdiff(fullSet,activeSet)) = residual'*X(:,mysetdiff(fullSet,activeSet)); % compute the correlations

C = max(abs(cor));
index = find( abs(abs(cor) - C) < epss); % find the current most correlating index
if C < epss % the proximal solution has been  found, it almost is zero.
    beta = zeros(p,1);
    return;
end
signn(index) = signn(index).*sign(cor(index))';
activeSet = [activeSet,index]; % updating active set
nonActiveSet = mysetdiff(fullSet,activeSet);
history.activeSet{1} = activeSet;

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
    history.beta_trace = [history.beta_trace, beta];
    history.correspondingBeta = correspondingBeta; history.correspondingBeta = [history.correspondingBeta, zeros(p,1)];
    history.activeSet{end} = activeSet; history.activeSet{end+1} = [];
    history.T = [history.T, norm(beta,1)];
    history.A = [history.A, A]; history.A = [history.A, 0];
    history.stopReason = '3';
    goIntoWhile = 0; % skip the following 'while' loop
else
    goIntoWhile = 1;
end

%% main loop
while goIntoWhile
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
    history.correspondingBeta = [history.correspondingBeta, correspondingBeta];
    history.A = [history.A, A];
    
    % To find out the next active predictor.
    cor = zeros(1,p);
    cor(nonActiveSet) = residual'*X(:,nonActiveSet); % compute the correlations between predictors and residual
    C = abs(residual'*X(:,activeSet(1))); % compute the biggest correlations
    if C < epss % the proximal solution has been found. Now it's time to return.
        goIntoWhile = 0; % This line is not necessary, just for readability.
        break;
    end
    cor_equi = zeros(1,p);
    cor_equi(nonActiveSet) = equiAngularVec'*X(:,nonActiveSet); % compute the correlations between predictors and the equiangular vector
    gammaLARS = Inf(2,p);
    gammaLARS(:,nonActiveSet) = [C - cor(nonActiveSet); C + cor(nonActiveSet)] ./ [A - cor_equi(nonActiveSet); A + cor_equi(nonActiveSet)];
    gammaLARS(gammaLARS<epss) = Inf; % 这行代码本来应该是“gammaLARS<0”的，但考虑到numerical error，才让“gammaLARS<epss”。以后如果程序出错，有可能是这里的问题，因为有可能epss的值设置的不合理。
    gammaLARSMin = min(min(gammaLARS(:,nonActiveSet))); % there may be more than one result. 比如说，刚好处在某两个向量的角平分线上，那么这两个的gamma值就是相等的
    gammaLASSO = zeros(1,p);
    gammaLASSO(activeSet) = -beta(activeSet) ./ correspondingBeta(activeSet);
    gammaLASSO(gammaLASSO<=epss) = Inf;
    gammaLASSOMin = min(gammaLASSO);
    if gammaLASSOMin < gammaLARSMin
        LASSOWorks = 1;
        gammaMin = gammaLASSOMin;
        indexToBeDeleted = find(gammaLASSO == gammaLASSOMin); % maybe there are one more index to be deleted. Be careful!
    else
        LASSOWorks = 0;
        gammaMin = gammaLARSMin;
        index1 = find( abs(gammaLARS(1,:)-gammaMin) < epss ); % find(gammaLARS(1,:) == gammaLARSMin);
        signn(index1) = 1;
        index2 = find( abs(gammaLARS(2,:)-gammaMin) < epss ); % find(gammaLARS(2,:) == gammaLARSMin);
        signn(index2) = -1;
        index = [index1, index2]; % 向前走的同时又能找到下一个(或多个)predictor，这两个步骤是同时完成的
    end
    
    if max(abs((residual-gammaMin*equiAngularVec)'*X(:,activeSet(1))))/p < epss % 如果这里的if语句能通过的话，说明residual已经几乎与当前的prediction正交，所以应该要停止计算beta并return了。如果不return，后面就会因为计算精度问题而导致错误结果
        % Now after the next step 'gammaLARSMin*equiAngularVec' is executed,
        % all the predictor will almost orthogonal to the residual. It
        % means that the proximal beta has been found, so we should return
        % there. If we do not finish this function there, some wrong
        % results will happen.
        beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
        prediction = X(:,activeSet)*beta(activeSet);
        residual = y - prediction;
        nonActiveSet = mysetdiff(fullSet,activeSet);
        history.beta_trace = [history.beta_trace, beta];
        history.correspondingBeta = [history.correspondingBeta, zeros(p,1)];
        history.activeSet{end+1} = activeSet;
        history.T = [history.T, norm(beta,1)];
        history.A = [history.A, 0];
        history.stopReason = '1';
        goIntoWhile = 0; % This line is not necessary, just for readability.
        break;
    end
    
    % Now the next active predictor has been found out, then we should make
    % prediction.
    beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
    if LASSOWorks
        beta(indexToBeDeleted) = 0; % To avoid numerical errors
    end
    prediction = X(:,activeSet)*beta(activeSet); % compute the current prediction vector
    residual = y - prediction; % compute the current residual
    if LASSOWorks
        activeSet = mysetdiff(activeSet, indexToBeDeleted); % delete the index from the active set
    else
        activeSet = [activeSet,index]; % now the new predictor is added to the active set
    end
    nonActiveSet = mysetdiff(fullSet,activeSet);
    history.beta_trace = [history.beta_trace, beta];
    history.activeSet{end+1} = activeSet;
    history.T = [history.T, norm(beta,1)];
    
    if isempty(nonActiveSet)
        % Now all of these predictors have been active, the prediction
        % vector should get to its final destination to finish its duty.
        % compute the equiangular vector
        signActive = signn(activeSet);
        correspondingBeta(activeSet) = XTX(activeSet,activeSet)\signActive; % inv(XTX(activeSet,activeSet))*signActive
        A = 1 / sqrt(signActive'*correspondingBeta(activeSet));
        correspondingBeta = A*correspondingBeta;
        equiAngularVec = X(:,activeSet)*correspondingBeta(activeSet);
        history.correspondingBeta = [history.correspondingBeta, correspondingBeta];
        history.A = [history.A, A];
        gammaLARSMin = residual'*equiAngularVec; % this is the projection length
        gammaLASSO = zeros(1,p);
        gammaLASSO(activeSet) = -beta(activeSet) ./ correspondingBeta(activeSet);
        gammaLASSO(gammaLASSO<=0) = Inf;
        gammaLASSOMin = min(gammaLASSO);
        if gammaLASSOMin < gammaLARSMin
            LASSOWorks = 1;
            gammaMin = gammaLASSOMin;
            indexToBeDeleted = find(gammaLASSO == gammaLASSOMin);
        else
            LASSOWorks = 0;
            gammaMin = gammaLARSMin;
        end
        beta(activeSet) = beta(activeSet) + gammaMin*(correspondingBeta(activeSet)); % let it step forward along all the active predictors
        if LASSOWorks
            beta(indexToBeDeleted) = 0; % in case of the machine numerical unprecision
        end
        prediction = X(:,activeSet)*beta(activeSet);
        residual = y - prediction;
        history.beta_trace = [history.beta_trace, beta];
        history.T = [history.T, norm(beta,1)];
        if LASSOWorks
            activeSet = mysetdiff(activeSet, indexToBeDeleted);
            nonActiveSet = mysetdiff(fullSet,activeSet);
            history.activeSet{end+1} = activeSet;
        else
            history.activeSet{end+1} = [];
            history.correspondingBeta = [history.correspondingBeta, zeros(p,1)];
            history.A = [history.A, 0];
            history.stopReason = '2';
            goIntoWhile = 0; % This line is not necessary, just for readability.
            break;
        end
    end
end

% compute Lambda based on t0
tempLength = length(history.T) - 1;
history.Lambda = zeros(1,tempLength+1);
history.Lambda(tempLength+1) = 0;
for ii = tempLength:(-1):1
    history.Lambda(ii) = history.A(ii)^2*(history.T(ii+1)-history.T(ii)) + history.Lambda(ii+1);
    % Why is the above formula? Please check equation (5.25) of the
    % reference paper.
end

%% compute the beta for the constraint 'T' and "Lambda"
if isempty(T) && isempty(Lambda)
    return;
end
clear beta;
beta.TLambdaPairs = [];

% solutions for T constraint
if ~isempty(T)
    tempLength = length(history.T) - 1;
    lengthT = length(T);
    beta.T = [];
    for ii = 1:lengthT
        if T(ii) < 0
            beta.T(:,ii) = NaN(p,1);
            beta.TLambdaPairs = [beta.TLambdaPairs, [T(ii);NaN] ];
            continue;
        end
        if T(ii) >= history.T(end)
            beta.T = [beta.T, history.beta_trace(:,end)];
            beta.TLambdaPairs = [beta.TLambdaPairs, [T(ii);0] ];
            continue;
        end
        for jj = tempLength:(-1):1
            if history.T(jj) > T(ii), continue; end
            tempLambda = history.Lambda(jj) - history.A(jj)^2*(T(ii)-history.T(jj));
            beta.TLambdaPairs = [beta.TLambdaPairs, [T(ii);tempLambda] ];
            betaTemp = history.beta_trace(:,jj) + (T(ii)-history.T(jj))*history.A(jj)*history.correspondingBeta(:,jj);
            beta.T = [beta.T, betaTemp];
            % Why is the above formula? Please check equation (5.17) of the
            % reference paper.
            break;
        end
    end
end

% solutions for Lambda constraint
if ~isempty(Lambda)
    tempLength = length(history.Lambda);
    lengthLambda = length(Lambda);
    beta.Lambda = [];
    for ii = 1:lengthLambda
        if Lambda(ii) < 0,
            beta.Lambda(:,ii) = NaN(p,1);
            beta.TLambdaPairs = [beta.TLambdaPairs, [NaN;Lambda(ii)] ];
            continue;
        end
        if Lambda(ii) >= history.Lambda(1)
            beta.Lambda = [beta.Lambda, zeros(p,1)];
            beta.TLambdaPairs = [beta.TLambdaPairs, [0;Lambda(ii)] ];
            continue;
        end
        for jj = tempLength:(-1):1
            if history.Lambda(jj) <= Lambda(ii), continue; end
            tempT = history.T(jj) + (history.Lambda(jj)-Lambda(ii))/history.A(jj)^2;
            beta.TLambdaPairs = [beta.TLambdaPairs, [tempT;Lambda(ii)] ];
            betaTemp = history.beta_trace(:,jj) + (history.Lambda(jj)-Lambda(ii))/history.A(jj)*history.correspondingBeta(:,jj);
            beta.Lambda = [beta.Lambda, betaTemp];
            % Why is the above formula? Please check equation (5.24) of the
            % reference paper.
            break; 
        end
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
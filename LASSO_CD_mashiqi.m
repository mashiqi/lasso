function [beta,history] = LASSO_CD_mashiqi(X,y,lambda,standardize)
%{
% 
% Coordinate Descent algorithm for LASSO problem.
% 
% 
%	AUTHOR      Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
%	DATE        1/24/2015
%	VERSION     1.0
% 
% 
% This function tries to find the proper solution of the following question:
%            argmin_(beta) 0.5*||y - X*beta||_2 + lambda*||beta||_1
% 
%
% INPUT ARGUMENTS:
%
%	X             - samples of predictors. Each column of X is a predictor,
%                   and each row is a data sample.
%
%	y             - the response.
%
%	lambda        - the coefficient of the norm-one term.
%
%	standardize   - the indicator. If standardize == 1, every column in X
%                   and y will be standardized to mean zero and standard
%                   deviation 1. And if its value is 0, then
%                   standardization process will not be executed.
%                   standardize == 0 as default. 
%
% OUTPUT ARGUMENTS:
%
%	beta          - weight vector.
% 
% 
% REFERENCE:
%
%	[1] https://drive.google.com/file/d/0B-WY1fXw1zaNLW13b0lDRGlvVXM/view
%}

%% parameter check
if nargin < 4 || isempty(standardize)
    standardize	= 0;
end
if nargin < 3 || isempty(lambda)
    lambda      = 0;
end
if ~isscalar(lambda)
    error('Only one constraint at a time.');
elseif lambda < 0
    error('Norm constraint is invalid, please check it.');
end
if nargin < 2
    error('Parameter invalid, please check it.');
end
if isvector(y) && (size(X,1) ~= size(y,1))
    error('Y is not a vector, or the length of Y is not equal to the number of row of X');
end

%% initialization
if standardize == 1
    X = bsxfun(@minus,X,mean(X,1));
    X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
    y = bsxfun(@minus,y,mean(y,1));
end
p               = size(X,2); % number of predictors
epss            = 1e-10;
fullSet         = 1:p;
activeSet       = false(1,p);
nonActiveSet	= true(1,p);
betaNew         = zeros(p,1);
columnStepSize  = 1000;
betaTrace       = zeros(p,columnStepSize);
Xnorm2          = sum(X.*X,1);
f               = zeros(1,columnStepSize);
fOld            = 0.5*norm(y,2)^2;
fNew            = fOld;
yMinusXBeta     = y;
maxIter         = 1e4;
iter            = 1;
FLAG            = true;

%% find the index of the entry with maximum gredient
[~,j]           = max(abs((y-X*betaNew)'*X));
activeSet(j)	= true;
nonActiveSet(j) = false;

%% coordinate decsent begins
while FLAG
    for index = fullSet(activeSet)
        %% update
        fOld            = fNew;
        index_c         = logical([1:(index-1),0,(index+1):p]); % this may be faster than "index_c = setdiff(1:p,index)"
        beta_index_old	= betaNew(index);
        temp1           = X(:,index)' * (y-X(:,index_c)*betaNew(index_c));
        beta_index_new	= wthresh(temp1,'s',lambda) ./ Xnorm2(index);
        betaNew(index)	= beta_index_new;
        
        % The following line may be a little bit complicated, but it is much
        % faster that just directly compute current f by:
        %                f=0.5*||y-Xbeta||_2^2+lambda||beta||_1.
        fNew = fOld	- yMinusXBeta'*X(:,index)*(beta_index_new-beta_index_old) ...
                    + 0.5*Xnorm2(index)*(beta_index_new-beta_index_old)^2 ...
                    + lambda*(abs(beta_index_new)-abs(beta_index_old));
        
        yMinusXBeta = yMinusXBeta - X(:,index)*(beta_index_new-beta_index_old);
        
       %% store the new result
        betaTrace(:,iter)	= betaNew;
        f(iter)             = fNew;
        if iter == columnStepSize
            columnStepSize	= columnStepSize + 1000;
            
            % increase the size of 'betaTrace'
            tempTrace       = betaTrace;
            betaTrace       = zeros(p,columnStepSize);
            betaTrace(:,1:(columnStepSize-1000))	= tempTrace;
            
            % increase the size of 'f'
            tempf           = f;
            f               = zeros(1,columnStepSize);
            f(1:(columnStepSize-1000))	= tempf;
            
            % clear temp variable
            clear tempTrace tempf;
        end
        
        %% checking if the maxmum iteration number has reached
        if iter >= maxIter
            FLAG = false;
            break;
        else
            iter = iter + 1;
        end
    end
    if ( abs(fOld-fNew) < epss )
        if isequal(fullSet,fullSet(activeSet))
            break;
        else
            % find the index of the entry with maximum gredient
            gredient                = -ones(1,p);
            gredient(nonActiveSet)	= abs((y-X(:,nonActiveSet)*betaNew(nonActiveSet))'*X(:,nonActiveSet));
            [~,j]                   = max(gredient);
            activeSet(j)            = true;
            nonActiveSet(j)         = false;
        end
    end
end

%% trim the unused columns
betaTrace(:,(iter+1):end)	= [];
f(:,(iter+1):end)           = [];

% get the final beta
beta                        = betaTrace(:,end);

% make results structured
history.betaTrace           = betaTrace;
history.f                   = f;
clear betaTrace f;
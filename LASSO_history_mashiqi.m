function beta = LASSO_history_mashiqi(history, T, Lambda)
%{
% LARS-LASSO algorithm, based on the output of function 'LASSO_mashiqi'.
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
%	history         - the output of function 'LASSO_mashiqi'.
% 
% 	T				- the norm-one type (1)constraint. T can be a vector. T = [] as default.
%
% 	Lambda			- the corresponding type (2) constraint. Lambda can be a vector.
% 					  Lambda = [] as default.
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
% 
% EXAMPLE:
% 
% This function, based on the result of function 'LASSO_mashiqi', quickly
% compute the results of constraints 'T' and 'Lambda'.
% related values. If you just want to get the specific beta for a specific
% constraint T, then this function maybe too cast in time spend. You can
% run the following code to get started:
%         	[beta,history] = LASSO_mashiqi(X, y);
%           beta2 = LASSO_history_mashiqi(history, T);
% where the variable T is the norm-one constraint and 'beta2.T' gives the
% lasso solution under this norm-one constraint T.
% There is a simple example:
%           clear;
%        	X = randn(200,50);
%         	beta = randn(50,1);
%         	y = X*beta;
%         	
%        	standardize = 0;
%        	[beta_hat1,lasso_history] = LASSO_mashiqi(X,y,T,[],standardize);
%           
%           T = [1 2];
%           Lambda = [0.1 0.2];
%           beta_hat2 = LASSO_history_mashiqi(lasso_history, T, Lambda);
%         	beta_hat2.T
%        	beta_hat2.Lambda
% That's it.
% 
% 
% REFERENCE:
% 
% [1] Efron, Bradley, et al. "Least angle regression." The Annals of 
%    statistics 32.2 (2004): 407-499.
%}

%% compute Lambda based on t0
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
        for jj = tempLength:(-1):1 % There can be improved by using binary search
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
        for jj = tempLength:(-1):1 % There can be improved by using binary search
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
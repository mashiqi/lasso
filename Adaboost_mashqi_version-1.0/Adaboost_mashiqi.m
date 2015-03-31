function [tree,beta,weight,trainingError,exponentialLoss] = Adaboost_mashiqi(X,y,options)
%{
%   Adaboost algorithm.
% 
% 
%	AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
%	DATE            - 1/28/2015
%	VERSION         - 0.7
% 
% 
% There should be some instructions. Coming soon!
% 
% 
% INPUT ARGUMENTS:
% 
%	X               - samples of predictors, every row is a sample
% 
%   y               - response to the X input. It should be a vertical
%                     vector, and every column is sample point
% 
%   options        	- optional settings. It is a structure that contatins following optional number
% 					  of parameter-value pairs to specify some, all, or none of the them: 
% 
% 
% OUTPUT ARGUMENTS:
% 
%	tree            - every column is a weak tree
% 
%   beta            - please refer to (10.12) of the reference book.
% 
%   weight          - please refer to (10.14) of the reference book.
% 
%   noEqualFlag 	- please refer to (10.10) of the reference book.
% 
%   trainingError 	- please refer to (10. 8) of the reference book.
% 
%   exponentialLoss	- please refer to (10. 8) of the reference book.
% 
% 
% EXAMPLE:
% 
%   nSample                 = 100;
%   nPredictor              = 10;
%   X                       = randn(nSample,nPredictor);
%   ChiSqure                = sum(X.^2,2);
%   y                       = 2*(ChiSqure > 9.342) - 1;
%   options = OptionsSettings('printFigure',true);
% 	[tree,beta,weight,noEqualFlag,trainingError] = ...
%                        	Adaboost_mashiqi(X,y,options);
% 
% 
% REFERENCE:
% 
%	[1] Hastie, Trevor, et al. The elements of statistical learning. Vol.
%	2. No. 1. New York: Springer, 2009.
%}

%% parameter check
if nargin < 2
    error('Error:XandYmissing','Arguments to Adaboost_mashiqi should at least contain ''X'' + ''y''!\n');
end
if ~(isvector(y) && ~isrow(y))
    error('Error:YFormatError','Argument y should be a vertical vector.\n');
end
if size(X,1) ~= size(y,1)
    error('Error:XandYmissing','Length of Argument y should be equal to the number of row of Argument X.\n');
end
% if rem(nargin,2)
%     error('Error:ParameterFormat','Arguments to Adaboost_mashiqi should be ''X'' + ''y'' + ''(property,value)-pairs''!\n');
% end

%% initialization
if nargin < 3
    options = OptionsSettings();
end
[nSample,nPredictor]	= size(X);
tree.index              = zeros(1,      options.treeNumber);
tree.threshold          = zeros(1,      options.treeNumber);
tree.direction          = zeros(1,      options.treeNumber);
beta                    = zeros(1,      options.treeNumber);
weight                  = zeros(nSample,options.treeNumber);
noEqualFlag             = false(nSample,options.treeNumber);
exponentialLoss        	= zeros(1,      options.treeNumber);
trainingError           = zeros(1,      options.treeNumber);
tree.treeNumber         = -1;

%% Adaboost algorithm begins
prediction  = 0;
weight(:,1)	= 1/nSample;
for iTree = 1:options.treeNumber
    if options.printInfo
        sprintf('treeNumber = %d \t iTree = %d\n',options.treeNumber,iTree)
   end
    %% compute the weak classificator
    errorValOpt	= Inf;
    for index = 1:nPredictor
        rangeMin = min(X(:,index));
        rangeMax = max(X(:,index));
        thresholdCandidate = linspace(rangeMin,rangeMax,options.stairNumber);
        thresholdCandidate(1) = thresholdCandidate(1) - 0.1;
        thresholdCandidate(end) = thresholdCandidate(end) + 0.1;
        for threshold = thresholdCandidate
            for direction = [-1,1]
                % make prediction
                nPrediction = RegressFunction(X(:,index),threshold,direction);
                noEqual = nPrediction ~= y;
                errorVal = sum(weight(noEqual,iTree));
                if errorVal < errorValOpt
                    % store temporary optimal values
                    errorValOpt           	= errorVal;
                    indexOpt              	= index;
                    thresholdOpt           	= threshold;
                    directionOpt           	= direction;
                    nPredictionOpt        	= nPrediction;
                    noEqualFlag(:,iTree)	= noEqual;
                end
            end
        end
    end
    % store nth classificator
    tree.index(iTree)       = indexOpt;
    tree.threshold(iTree)	= thresholdOpt;
    tree.direction(iTree)	= directionOpt;
    
    %% update some parameters
    % update beta. Please refer to the (10.12) of the reference book.
%     nError      = sum(weight(noEqualFlag(:,nTree),nTree)) / sum(weight(:,nTree));
    beta(iTree)	= log((1-errorValOpt) / errorValOpt);

    % update weight. Please refer to the (10.14) of the reference book.
    weight( noEqualFlag(:,iTree),iTree+1)	= weight( noEqualFlag(:,iTree),iTree) .* exp(beta(iTree));
    weight(~noEqualFlag(:,iTree),iTree+1)	= weight(~noEqualFlag(:,iTree),iTree);
    weight(:,iTree+1)                       = weight(:,iTree+1)/sum(weight(:,iTree+1));
    %{
    % normalize weight. It do not give theoretical effect on the algorithm
    % results, but doing itjust in case of numerical error cased by too
    % large or too small scalar arithmetic.
    %}
    
    % make prediction
    prediction             	= prediction + beta(iTree)*nPredictionOpt;
    trainingError(iTree)  	= sum(sign(prediction) ~= y)/nSample;
    exponentialLoss(iTree)	= sum(exp(-y.*prediction))/nSample;
    
    %% plot
    %-- for debug. The following several line can be deleted
    N = 10;
    if options.printFigure && (mod(iTree,N) == 0) && (iTree > N)
        figure(1);
        % plot classification error
        subplot(2,1,1),plot((iTree-N):iTree,  trainingError((iTree-N):iTree),'b');hold on;
        xlabel('Number of Adaboost weak trees');ylabel('Classification Error');title('Adaboost algorithm error curves');
        
        % plot exponential loss
        subplot(2,1,2),plot((iTree-N):iTree,exponentialLoss((iTree-N):iTree),'b');hold on;
        xlabel('Number of Adaboost weak trees');ylabel('Exponential Loss');title('Adaboost algorithm error curves');
        pause(0.01);
    end
    %-- for debug. The above several line can be deleted
    
    %% stop criteria
    if iTree > 11
        % use the recent average total error to decide if it's time to stop
        % iteration.
        if mean(abs(exponentialLoss((iTree-10):iTree))) < options.epsTolerance
            options.stopReason              = 1;
            % trim unused memory
            tree.index((iTree+1):end)      	= [];
            tree.threshold((iTree+1):end)	= [];
            tree.direction((iTree+1):end)	= [];
            beta((iTree+1):end)             = [];
            weight(:,(iTree+1):end)        	= [];
            noEqualFlag(:,(iTree+1):end)	= [];
            trainingError((iTree+1):end)	= [];
            exponentialLoss((iTree+1):end)	= [];
            break;
        end
    end
end

%% final settings
tree.treeNumber         = length(tree.index);
if (iTree == options.treeNumber) && isempty(options.stopReason)
    options.stopReason  = 2;
end
if options.printInfo % to print information
    switch options.stopReason
        case 1, % convergence reached before maxIters
            fprintf('Info: Algorithm converges in %d weak trees because there is no improvements in total error.\n\n', iTree);
        case 2, % maxIters reached
            fprintf('Info: Maximum weak tree number %d has reached, and there is no evidence showing that total error has converged.\n\n', options.treeNumber);
        otherwise,
        	warning('MATLAB:UnexpectedError','Info: Unexpected error.\n\n');
    end
end
end
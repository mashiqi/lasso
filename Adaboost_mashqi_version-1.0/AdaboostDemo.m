%{
%   Adaboost algorithm Demo
% 
% 
%	AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
%	DATE            - 1/30/2015
%	VERSION         - 0.7
% 
% 
% This is a demo file to show you how to use the 'Adaboost_mashiqi'
% function, and exhibit the test comparison of this Adaboost algorithm. At
% the beginning of this file, is a FLAG 'YesIHaveTrainingResult' to let you
% choose to run the training process or just directly load the training
% results. In doing so, you can at the first time run the training process
% and save these variables you'v gotten from this training process, then
% every time you run this file, you can just directly load these variables
% you have gotten in the first time, so save your time.
% 
% In the Adaboost training function 'Adaboost_mashiqi', it use the
% instruction 'Algorithm 10.1' in page 339 of the reference book. This
% function use the 'single node decision tree' as the weak classifier.
% 
% In the Adaboost testing process, this demo will plot two figure: the
% figure on the left of the screen exhibit the classification results, and
% the figure on the right shows the error curve comparison of both
% classified error rate and exponential error on both training and testing
% data set. These two figures shows dynamically with the nth weak tree
% being added to the active set: in the first, just the first weak tree is
% used to make prediction and results will be plotted, and then the second
% weak tree will be active to jointly make prediction with the first one,
% and results also will be shown. And so on, every time a new tree will be
% active to make the current predition more accurate, and these two figure
% will be updated due to the effect of this added tree. This will be going
% on and on until the final tree is concluded to jointly make prediction.
% For the speed consideration, the figure on the right will be updated
% every N times of weak tree adding process, where N, in the first line of
% the main body, can be changed by your own.
% 
% Enjoy it and have fun!
% 
% 
% REFERENCE:
% 
%	[1] Hastie, Trevor, et al. The elements of statistical learning. Vol.
%	2. No. 1. New York: Springer, 2009.
%}

clear; close all;
%% parameters you can changed
N                           = 10; % update figure_2 every N iterations, it MUST be integer!
YesIHaveTrainingResult      = false; % can only be set to 'true' or 'false'
NoIHaventTrainingResultYet  = ~YesIHaveTrainingResult;
useCppMexFile               = true;
%% ------------------------ Adaboost training process BEGIN ------------------------ %%
if NoIHaventTrainingResultYet
    %% generate synthetic data
    rand ('state',1000); % rng(1000,'v5uniform');
    randn('state',1000); % rng(1000,'v5normal');
    nSample            	= 3000;
    nPredictor         	= 10;
    X                  	= randn(nSample,nPredictor);
    ChiSqure           	= sum(X.^2,2);
    y                  	= 2*(ChiSqure > 9.342) - 1;

    %% separate synthetic data into training and testing parts
    numSmpTraining      = ceil(2/3*nSample);
    numSmpTesting       = nSample - numSmpTraining;
    trainingIndexes     = sort(randsample((1:nSample)',numSmpTraining), 'ascend');
    testingIndexes      = setdiff((1:nSample)',trainingIndexes);
    XTraining           = X(trainingIndexes,:);
    yTraining           = y(trainingIndexes);
    XTesting            = X(testingIndexes,:);
    yTesting            = y(testingIndexes);

    %% Adaboost algorithm
	options = OptionsSettings('printInfo',true,'printFigure',false,'treeNumber',1000,'epsTolerance',1e-4);
    if useCppMexFile
        [tree,beta,weight,trainingError,exponentialLoss] = Adaboost_mashiqi_cpp(XTraining,yTraining,options);
    else
        [tree,beta,weight,trainingError,exponentialLoss] = Adaboost_mashiqi(XTraining,yTraining,options);
    end
    save trainingData   tree beta weight trainingError exponentialLoss nSample nPredictor X ChiSqure y...
                        numSmpTraining numSmpTesting trainingIndexes testingIndexes XTraining yTraining XTesting yTesting
%     saveas(1,'training_error_curves','fig');
end
%% ------------------------ Adaboost training process   END ------------------------ %%


%% ------------------------ Adaboost  testing process BEGIN ------------------------ %%
%% preparing for figure exhibition
%close all;
scrsz	= get(0,'ScreenSize');
figure1 = figure('Position',[scrsz(3)*03/40	scrsz(4)/3 scrsz(3)*2/5 scrsz(4)/2]);
figure2 = figure('Position',[scrsz(3)*21/40 scrsz(4)/3 scrsz(3)*2/5 scrsz(4)/2]);

%% Initialization
if YesIHaveTrainingResult
    load('trainingData');
end
prediction      = zeros(numSmpTesting,tree.treeNumber);
testingError    = zeros(1,tree.treeNumber);
testingExpLoss	= zeros(1,tree.treeNumber);

%% At the beginning, plot the training errors
figure(figure2);
% plot classification error
subplot(2,1,1),trainingClaErrorHandle = plot(1:tree.treeNumber,      trainingError,'b');
xlabel('Number of Adaboost weak trees');ylabel('Classification Error');title('Adaboost algorithm error curves');
hold on;

% plot exponential loss
subplot(2,1,2),trainingExpErrorHandle = plot(1:tree.treeNumber,    exponentialLoss,'b');
xlabel('Number of Adaboost weak trees');ylabel('Exponential Loss');title('Adaboost algorithm error curves');
hold on;

%% begin to make prediction in the training set
for nTree = 1:tree.treeNumber
    %% classification and error computing
    prediction(:,nTree)     = RegressFunction(XTesting(:,tree.index(nTree)),  tree.threshold(nTree),  tree.direction(nTree)   );
    betaPrediction          = prediction(:,1:nTree)*beta(1:nTree)';
    testingError(nTree)  	= sum(sign(betaPrediction) ~= yTesting)/numSmpTesting;
    testingExpLoss(nTree)	= sum(exp(-yTesting.*betaPrediction))/numSmpTesting;
    
    %% plot classification results
    figure(figure1);
    INDEX1  = sort(find(sign(betaPrediction) == yTesting)); % samples that are correctly classified
    INDEX2  = sort(find(sign(betaPrediction) ~= yTesting)); % samples that are   falsely classified
    plot(   INDEX1,betaPrediction(INDEX1),'bo',...
            INDEX2,betaPrediction(INDEX2),'ro');
    legend('Correct','False');
    str = sprintf(  'Classification results on the testing set, with %d weak trees. \n\t ErrorRate = %.2f, ExpLoss = %.1f',...
                    nTree,testingError(nTree),testingExpLoss(nTree));
    xlabel('Index of testing sample points');ylabel('Prediction value');title(str);
    plotBoundary = max([abs(floor(min(betaPrediction)/3)*3), abs(ceil(max(betaPrediction)/3)*3)]);
    axis([0 1000 -plotBoundary plotBoundary]); % make figure frame fixed
    line([0,1000],[0,0],'linestyle',':','Color','k'); % base line
    pause(0.05); % leave a time gap to let figure shown
    
    %% plot Classification Error Rate and Exponential Loss
    if (mod(nTree,N) == 0) && (nTree > N)
        figure(figure2);
        % plot classification error
        subplot(2,1,1),testingClaErrorHandle = plot((nTree-N):nTree,  testingError((nTree-N):nTree),'r');
        xlabel('Number of Adaboost weak trees');ylabel('Classification Error');title('Adaboost algorithm error curves');
        legend([trainingClaErrorHandle,testingClaErrorHandle],'Training','Testing');
        hold on;
        
        % plot exponential loss
        subplot(2,1,2),testingExpErrorHandle = plot((nTree-N):nTree,testingExpLoss((nTree-N):nTree),'r');
        xlabel('Number of Adaboost weak trees');ylabel('Exponential Loss');title('Adaboost algorithm error curves');
        legend([trainingExpErrorHandle,testingExpErrorHandle],'Training','Testing','Location','North');
        hold on;
        pause(0.05);
    end
end
1; % you can set a breakpoint here
%% ------------------------ Adaboost  testing process   END ------------------------ %%
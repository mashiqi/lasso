function options = OptionsSettings(varargin)
%{
%   OptionsSettings function for Adaboost algorithm.
% 
% 
%	AUTHOR          - Shiqi Ma (mashiqi01@gmail.com, http://mashiqi.github.io/)
%	DATE            - 1/28/2015
%	VERSION         - 0.7
% 
% 
% INPUT ARGUMENTS:
% 
%	Optional number of parameter-value pairs to specify some, all, or none
%	of the them: 
% 
%	maxIteration    - the maximum number of algorithm iterations
%	stairNumber     - stair number
%	treeNumber      - maximum tree number
%	epsTolerance    - the tolerance threshold
%	printInfo       - flag indicating whether to show algorithm's detail
%                     information
%   printFigure     - flag indicating whether to show figures
%   stopReason      - algorithm stop reasons:
%                           0 - initial value;
%                           1 - convergence reached;
%                           2 - maxIteration reached;
% 
% OUTPUT ARGUMENTS:
% 
%	options        	- response to the X input. It should be a vertical
%                     vector, and every column is sample point
% 
% 
% EXAMPLE:
%   options1 = OptionsSettings(); % default settings
%   options2 = OptionsSettings('printFigure',true,...
%                              'treeNumber',1000,...
%                              'epsTolerance',1);
%}

%% these following parameters should always have values
% the maximum number of algorithm iterations
options.maxIteration            = 10000;

% stair number
options.stairNumber             = 50;

% maximum tree number
options.treeNumber              = 1e3;

% the tolerance threshold
options.epsTolerance            = 1e-4;

% flag indicating whether to show algorithm's detail information
options.printInfo               = true;

% flag indicating whether to show figures
options.printFigure             = false;

% algorithm stop reasons:
%       0 - initial value;
%       1 - convergence reached;
%       2 - maxIteration reached;
options.stopReason              = 0;

%% addtional paramter setting
if isempty(varargin)
    return;
end
numSettings	= length(varargin)/2;
if numSettings == 0 % options setting over!
    return;
end
for n = 1:numSettings
    property_	= varargin{2*n-1};
    value       = varargin{2*n  };
    switch upper(property_)
        case 'MAXITERATION',
             % the maximum number of EM algorithm iterations
            options.maxIteration        = value;
            
        case 'STAIRNUMBER',
            % stair number
            options.stairNumber         = value;
            
        case 'TREENUMBER',
            % maximum tree number
            options.treeNumber          = value;
            
        case 'EPSTOLERANCE',
            % the tolerance threshold
            options.epsTolerance        = value;
            
        case 'PRINTINFO',
            % flag indicating whether to show algorithm's detail information
            options.printInfo           = value;
            
        case 'PRINTFIGURE',
            % flag indicating whether to show figures
            options.printFigure       	= value;
        case 'STOPREASON',
            % algorithm stop reasons:
            %       1 - convergence reached;
            %       2 - maxIteration reached;
            options.stopReason          = value;
            
        otherwise,
            % Unrecognised initialization property
            error('Error:ParameterUnknown','Unrecognised initialization property: ''%s''\n', property_);
    end
end
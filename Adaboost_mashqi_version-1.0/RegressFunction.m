function yHat = RegressFunction(X,threshold,direction)
%{
%	X        	- should be a vertical vector
%	threshold	- should be a scalar or horizontal vector that is the same
%                 length with the numer of column of X
%	direction	- should be either -1 or 1
%}

%% parameter check
if isempty(X) || isempty(threshold)
    yHat = 0;
    return;
end
if ~(isvector(threshold) && isrow(threshold))
    error('Error:YFormatError','Argument y should be a horizontal vector.\n');
end
if ~isscalar(threshold) && (size(X,2) ~= size(threshold,2))
    error('Error:ColumnNumberNotEqual','Column number of Argument X and Argument threshold should be the same.\n');
end
if isempty(direction)
    yHat = zeros(size(X));
    return;
end

%% regression
if isscalar(threshold)
    % when there is only one thershold, use the following code. It is
    % faster than bsxfun(@gt/@lt,X,threshold).
    if direction == 1
        yHat = logical(X > threshold);
    elseif direction == -1
        yHat = logical(X < threshold);
    else
        error('Error:Value','variable ''direction'' has a invalid value.\n');
    end
    yHat = 2*yHat - 1;
    return;
else
    % when encounter with multiple thersholds, bsxfun(@gt/@lt,X,threshold)
    % is faster.
    if direction == 1
        yHat = bsxfun(@gt,X,threshold);
    elseif direction == -1
        yHat = bsxfun(@lt,X,threshold);
    else
        error('Error:Value','variable ''direction'' has a invalid value.\n');
    end
    yHat = 2*yHat - 1;
    return;
end
end
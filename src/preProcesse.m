function [Xtrain, Xtest, bias, scale] = preProcesse(Xtrain, Xtest)

min_ = min(Xtrain);
max_ = max(Xtrain);
bias = (min_ + max_) / 2;
scale = max_ - bias;

% centralize, rescale
Xtrain = bsxfun(@rdivide, bsxfun(@minus, Xtrain, bias), scale); % ??? normorlize data into -1 and +1, but why not 0 to 1
Xtest = bsxfun(@rdivide, bsxfun(@minus, Xtest, bias), scale);

% make sure NO MISSING VALUES
assert( all(isfinite(Xtrain(:))));
assert( all(isfinite(Xtest(:))));

% fix missing value, replace it with mean value
% mean_ = repmat(nanmean(Xtrain), size(Xtrain, 1), 1); % ???
% Xtrain(isnan(Xtrain)) = mean_(isnan(Xtrain));
% Xtest(isnan(Xtest)) = mean_(isnan(Xtest));

% now min(Xtrain) = -1, max(Xtrain) = +1
end
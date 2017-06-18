function [ param ] = trainRBFNN(Xtrain, Ytrain, param)
% Xtrain shound be pre-processed
if isfield(param, 'centerCache')
    param.rngState = param.centerCache.rngState;
    U = param.centerCache.U;
else
    param.rngState = rng; % save rng before kmeans
    [U, ~] = findCenter(Xtrain, Ytrain, param.nCenter);     % nCenter: M
end
assert(same(size(U), [param.nCenter, size(Xtrain, 2)] ) == 2);
% compute width for hidden neurons
% use mean distance between centers (hezhimin, see also width_Mean.m)

% for debug
pd = pdist(U);

V = repmat(mean(pdist(U)), param.nCenter, 1);
% re-scale with param.alpha
V = V * param.alpha;

% according to the paper
% f(x) = Sum ( w_j * exp ^ ( - ||x - u_j||^2 / (2 * v_j^2) ) )
% note that there's a '2' before v^2
t1 = pdist2(Xtrain, U).^2; % ||x - u_j|| ^2

% % for debug
% radius = (- 2 * (V.^2))';
% tttttt = bsxfun(@rdivide, t1, (- 2 * (V.^2))');
% a = sum(radius); 
% b = sum(tttttt); 

t2 = exp(bsxfun(@rdivide, t1, (- 2 * (V.^2))')); % phi(x)

% LS regression
% min ||t2 * W - Ytrain||^2
W = lscov(t2, Ytrain);

param.U = U;
param.V = V;
param.W = W;
assert(same(size(param.U), [param.nCenter, size(Xtrain, 2)]) == 2);
assert(same(size(param.V), [param.nCenter, 1]) == 2);
assert(same(size(param.W), [param.nCenter, size(Ytrain, 2)]) == 2);
end


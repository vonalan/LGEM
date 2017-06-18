function [U, centerLabel] = findCenter(Xtrain, Ytrain, nCenter)

if size(Ytrain, 2) == 1 % two-class case
    Ytrain = [Ytrain == 1, Ytrain == -1]; % concatenating horizontally and converting to 0-1 vector
end

Ytrain = (Ytrain == 1);
assert(all(sum(Ytrain, 2) == ones(size(Xtrain, 1), 1))); % make sure no missing data

cn = centerNum(nCenter, sum(Ytrain, 1)); % number of clusters of each class
U = zeros(nCenter, size(Xtrain, 2));
centerLabel = zeros(nCenter, 1);

idx2 = 0;
for i = 1:numel(cn) % ???
    idx1 = idx2 + 1;
    idx2 = idx2 + cn(i);
    centerLabel(idx1:idx2) = i;
    % clustering INSIDE each class
    % then put all centers together
    
    % for debug
    [~, centroids, sumd] = kmeans(Xtrain(Ytrain(:, i), :), cn(i), 'EmptyAction', 'singleton');
    
    [~, U(idx1:idx2, :)] = ... % codes are segment to two lines 
        kmeans(Xtrain(Ytrain(:, i), :), cn(i), 'EmptyAction', 'singleton');
end
assert(idx2 == nCenter);
% done
end


function [ Y ] = label2vector(label)

assert(size(label, 2) == 1);
uniqLabel = unique(label(:)); % and sorted
Y = zeros(size(label, 1), numel(uniqLabel));

for i = 1:numel(uniqLabel)
    Y(:, i) = (label == uniqLabel(i)); % ???
end

assert(same(sum(bsxfun(@times, uniqLabel', Y), 2), label) == 2);

if numel(uniqLabel) == 2
    Y = Y(:, 1) - Y(:, 2); % convert to +1/-1
end

end

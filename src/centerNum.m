function [ cn ] = centerNum(nCenter, nSample)

assert(size(nSample, 1) == 1);
assert(nCenter >= size(nSample, 2));
assert(nCenter <= sum(nSample));

cn = ones(size(nSample));
nCenter = nCenter - sum(cn); %??? the reason why nCenter should be larger than number of class, but why? 
for i = 1:nCenter
   [~, j] = max(nSample ./ cn); % ??? why? rescale? 
   cn(j) = cn(j) + 1;
end
end


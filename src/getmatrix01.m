function res = getmatrix01(m)
res = zeros(size(m));

if size(m,2) == 1
    res = [m>0];
else
for i = 1:size(m,1)
    [~,index] = find(m(i,:) == max(m(i,:))); % ???
    res(i,index) = 1;
end

end

end
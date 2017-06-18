function [ output ] = testRBFNN( Xtest, param )
% compute RBFNN output

t1 = pdist2(Xtest, param.U).^2;
t2 = exp(bsxfun(@rdivide, t1, (- 2 * (param.V.^2))'));
output = t2 * param.W;

end


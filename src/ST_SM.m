%%% SM computation;
%%% independent of classifier
%%% first used in MLPNN

%%% to calculate deltaX, generate 1000 feature points randomly around
%%% training samples with neighbor Q
function SM_vector = ST_SM(train_sample, W, U, V, deltaX) % deltaX is fluctuatio of trainX
    [~,N] = size(train_sample);
    SM_vector = zeros(1, N);
    H = size(deltaX,1);
    tmpsum = 0;
    t1 = pdist2(train_sample,U).^2;
    t2 = exp(bsxfun(@rdivide, t1, (-2 * (V .^ 2))'));
    f1 = t2 * W;
    for i = 1:H
        

        t1 = pdist2(bsxfun(@plus,train_sample, deltaX(i,:)),U).^2;
        t2 = exp(bsxfun(@rdivide, t1, (-2 * (V .^ 2))'));
        f2 = t2 * W;
    
        tmpsum = tmpsum + (f2 - f1) .^ 2 ;
    end
    SM_vector = tmpsum / H;
end
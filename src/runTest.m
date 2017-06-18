function [ param ] = runTest(param)

% load data
data = matfile(param.dataFileName);
X = data.X;
Y = data.Y;

nFeature = size(X, 2);
nClass = size(Y, 2);
nRound = param.nRound;
nCenterList = param.nCenterList;
alphaList = param.alphaList;
Q = param.Q;

% 0/1 vector representation
if nClass == 1 % two-class case
    Y01 = [Y == 1, Y == -1];
    assert(same(Y01 * [1; -1], Y) == 2);
    nClass = 2;
else
    assert(nClass > 2);
    Y01 = (Y == 1);
    assert(same(Y01, Y) == 2);
    assert(same(sum(Y01, 2), ones(size(Y01, 1), 1)) == 2);
end
assert(islogical(Y01));

% compute size of training set and testing set
nSample = sum(Y01, 1);
nTest = round(nSample .* param.testRatio);
nTrain = nSample - nTest;
assert(all(nTest(:) > 0));
assert(all(nTrain(:) > 0));
assert(sum(nSample(:)) == size(X, 1));
assert(size(X, 1) == size(Y01, 1));
assert(all(nCenterList(:) >= size(nTrain, 2)));
assert(all(nCenterList(:) <= sum(nTrain)));

% generate index for tests
% for each round, randomly seperate samples into testing set and training
% set
% save random seed
param.rngState = rng;

% trainMask(:, i), testMask(:, :) is the mask for ith round
% trainMask = false(size(X, 1), nRound);
trainMask = [true(500,1);false(500,1)]; 
% for j = 1:nClass
%     for i = 1:nRound
%         trainMask(Y01(:, j), i) = (randperm(nSample(j)) <= nTrain(j));
%     end
% end
testMask = ~ trainMask;
% param.trainMask = trainMask;
% param.testMask = testMask;

% matlab cell is column major ordered
%                     k,               j,              i
result = cell(numel(alphaList), numel(nCenterList), nRound);
for i = 1:nRound
    for j = 1:numel(nCenterList)
        centerCache = struct();
        for k = 1:numel(alphaList)
            caseParam = {};
%             i
%             j
%             k
            caseParam.nCenter = nCenterList(j);
            caseParam.alpha = alphaList(k);
            
            % prepare data
            Xtrain = X(trainMask, :);
            Xtest = X(testMask, :);
            Ytrain = Y(trainMask, :);
            Ytest = Y(testMask, :);            
            
            % preprocessing
%             [Xtrain, Xtest, ~, ~] = ...
%                 preProcesse(Xtrain, Xtest);

            % train
            if k > 1
                caseParam.centerCache = centerCache;
            end
            caseParam = trainRBFNN(Xtrain, Ytrain, caseParam);
            % save cache
            centerCache.U = caseParam.U;
            centerCache.rngState = caseParam.rngState;
            % test
            outputTrain = testRBFNN(Xtrain, caseParam);
            outputTest = testRBFNN(Xtest, caseParam);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             outputTest01 = getmatrix01(outputTest);
%             Ytest01 = [Ytest == 1];
%             resTest = [outputTest01 == Ytest01];
%             testAcc = sum(sum(resTest))/numel(resTest);
            [~,oidx1] = max(outputTest,[],2); 
            [~,lidx1] = max(Ytest,[],2);
            idx1 = oidx1 == lidx1;
            testAcc = sum(idx1)/numel(idx1); 
            caseParam.testAcc = testAcc;

            % 
            [~,oidx2] = max(outputTrain,[],2); 
            [~,lidx2] = max(Ytrain,[],2);
            idx2 = oidx2 == lidx2;
            trainAcc = sum(idx2)/numel(idx2); 
            
            % aaa2 = [outputTrain(a,:),Ytrain(a,:)]; 
            
%             outputTrain01 = getmatrix01(outputTrain);
%             Ytrain01 = [Ytrain == 1];
%             resTrain = [outputTrain01 == Ytrain01];
%             
%             fm = sum(sum(resTrain));
%             fn = numel(resTrain);
%             trainAcc = sum(sum(resTrain))/numel(resTrain);
            caseParam.testAcc = trainAcc;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Remp, mean squared error
            % please note that the error is SQUARED
            RempTrain = mean((outputTrain - Ytrain).^2, 1);
            RempTest = mean((outputTest - Ytest).^2, 1);
            assert(2 == same(size(RempTrain), size(RempTest)));
            assert(2 == same(size(RempTrain), [1, size(Y, 2)]));
            caseParam.RempTrain = RempTrain;
            caseParam.RempTest = RempTest;
            % STSM
            caseParam = STSM(caseParam, Xtrain, Q);
            % part of formular (7)
            % RsmScore = sqrt(Remp) + sqrt(ESQ)
            % since error is computed in Euclidean function
            % in a multi-class problem,
            % squared error on every output dimension are summarize
            assert(size(RempTrain, 2) == size(caseParam.STSM, 2));
            caseParam.RsmScore = sqrt(sum(RempTrain, 2)) + sqrt(sum(caseParam.STSM, 2));
            assert(2 == same(size(caseParam.RsmScore), size(Q)));
            %
            caseParam.trainMask = trainMask(:, i);
            caseParam.testMask = testMask(:, i);
            % done
            result{k, j, i} = caseParam;
        end
    end
end

param.result = result;
end


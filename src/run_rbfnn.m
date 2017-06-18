% experiment settings
path = 'miku'; 
nRound = 1; 
nCenterList = (80:80); 
alphaList = logspace(0,0,1)';
ratio = 0.5; 
Q = logspace(-1,-1,1)';


% load data
data = matfile(path);
X = data.X;
Y = data.Y;
% Xtrain; 
% Xtest; 
% Ytrain; 
% Ytest; 


nFeature = size(X, 2);
nClass = size(Y, 2);


result = cell(numel(alphaList), numel(nCenterList), nRound);
for i = 1:nRound
    for j = 1:numel(nCenterList)
        centerCache = {};
        for k = 1:numel(alphaList)
            caseParam = {};
            caseParam.nCenter = nCenterList(j);
            caseParam.alpha = alphaList(k);
            
            % preprocessing
            [Xtrain, Xtest, ~, ~] = auto_scale(Xtrain, Xtest);

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

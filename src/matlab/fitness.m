function [ O ] = fitness(S)
    % calculating fitness for population; 
    
    global centroids; 
    
    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    % matlabpool local 4; 
    for i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d... \n',i, size(S,1)); 
        
        % mask 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 

        % flag = {0:not used, 1:train, 2:testa}; 
        [x_train, x_testa] = cluster(mcentroids); 
        
        % train and validate classifier
        [param] = classifier_light(c_train, y_train, x_train, c_testa, y_testa, x_testa); 
        [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param); 
        
        fprintf('acc_train: %.4f, acc_testa: %.4f, err_train: %.4f, err_testa: %.4f\n', ...
           acc_train, acc_testa, err_train, stsm_train); 
        
        O(i,:) = [err_train, stsm_train, acc_train, acc_testa]; 
        % break;
    end; 
    % matlabpool close; 
end


function [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param)
    rbfnnC = param.rbfnnC; 
    
    acc_train = rbfnnC.acc_train;
    acc_testa = rbfnnC.acc_testa;
    err_train = mean(rbfnnC.err_train);
    % err_testa = mean(rbfnnC.err_testa);
    stsm_train = mean(rbfnnC.stsm_train);
    % lgem_train = mean(rbfnnC.lgem_train);
end 
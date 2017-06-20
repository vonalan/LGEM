function [ O ] = fitness(S)
    % calculating fitness for population; 
    
    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    cates_dir = '../hmdb51_org_stips';
    % cates_raw = dir(cates_dir);
    cates_raw = ls(cates_dir);
    cates = cates_raw(3:end,:); 
    
    round = 1; 
    flag = 1; 
    C = size(cates,1); 
    K = 128; 
    
    centroids_path = ['../data/centroids_r', int2str(round), '_f', int2str(flag), '_k', int2str(K), '.txt']; 
    fprintf(centroids_path);
    fprintf('\n');
    centroids = importdata(centroids_path);
    
    % global c_train y_train x_train c_testa y_testa x_testa;
    % matlabpool local 4; 
    for i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d... \n',i, size(S,1)); 
        
        % mask 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 

        % flag = {0:not used, 1:train, 2:testa}; 
        [c_train, y_train, x_train] = kmeans_transform(cates, round, 1, C, mcentroids); 
        [c_testa, y_testa, x_testa] = kmeans_transform(cates, round, 2, C, mcentroids); 
%         x_train_path = '../data/x_train.txt'; 
%         x_testa_path = '../data/x_testa.txt'; 
%         y_train_path = '../data/y_train.txt'; 
%         y_testa_path = '../data/y_testa.txt'; 
% 
%         x_train = importdata(x_train_path);
%         x_testa = importdata(x_testa_path); 
%         y_train = importdata(y_train_path); 
%         y_testa = importdata(y_testa_path); 
%         
%         x_train = x_train(:,60:end); 
%         x_testa = x_testa(:,60:end);
        
        
        % train and validate classifier
        [param] = classifier_light(c_train, y_train, x_train, c_testa, y_testa, x_testa); 
        [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param); 
        
        fprintf('acc_train: %.4f, acc_testa: %.4f, err_train: %.4f, err_testa: %.4f\n', ...
           acc_train, acc_testa, err_train, stsm_train); 
        
        O(i,:) = [err_train, stsm_train, acc_train, acc_testa]; 
        %¡¡break;
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


function [cline, label, bovfs] = kmeans_transform(cates, round, flag, C, centroids)
    K = size(centroids,1);
    
    cline = zeros(0,1); % cline may be helpful someday; 
    label = zeros(0,C); 
    bovfs = zeros(0,K); 
    
    % cates = {'brush_hair'};
    % splitdir = '../data';
    % stipdir = '../data'; 
    
    split_dir = '../testTrainMulti_7030_splits'; 
    stip_dir = '../hmdb51_org_stips';
    
    for j = 1 : size(cates,1)
        cate = strcat(cates(j,:),'');
        split_path = [split_dir,'/',cate,'_test_split',int2str(round),'.txt']; 
        % fprintf(split_path);
        % fprintf('\n'); 
        
        split_set = importdata(split_path); 
        vname_set = split_set.textdata;
        mask_set = split_set.data; 

        for k = 1 : size(mask_set,1)
            vname = vname_set{k,1};
            mask = mask_set(k,1); 

            if mask == flag
                stip_path = [stip_dir,'/',cate, '/', vname,'.txt']; 
                % stip_path = '../data/brushing_hair_brush_hair_f_nm_np2_ba_goo_2.avi.txt'; 
                %¡¡fprintf(stip_path);
                % fprintf('\n');
                
                % [c,s] = read_stip_file(stip_path);
                stips_set = importdata(stip_path); 
                % stips_set = importdata('../hmdb51_org_stips/smile/show_your_smile_-)_smile_h_nm_np1_fr_med_0.avi.txt'); 
                stips = stips_set.data; 
                
                %cline
                s = zeros(1,K); 
                c = 0;
                
                %stips
                if size(stips,1) >= 0 && size(stips,2) == 169
                    c = size(stips,1);
                    
                    s = stips(:,8:end);
                    s = knnsearch(centroids, s); 
                    s = build_bow(c, s, K);
                end;
                
                % label
                l = zeros(1,C); 
                l(1,j) = 1;
                
                % need to be optimized!
                cline = [cline;c]; 
                label = [label;l];
                bovfs = [bovfs;s];
            end; 
            % break;
        end; 
        % break;
    end; 
    fprintf('%d, %d, %d',size(cline,1), size(label,1), size(bovfs,1)); 
    fprintf('\n');
end 


function [bovfs]  = build_bow(cline, xdata, nbins)
    bovfs = zeros(1, nbins); 
    hist = tabulate(xdata)'; % the upper bound of hist may not be nbins, but the maximum of xdata;
    bovfs(1,hist(1,:)) = hist(2,:); 
end 

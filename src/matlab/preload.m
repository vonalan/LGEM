function [C] = preload( round, K, ns1, ns2)
    global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;
    
    cates_dir = '../hmdb51_org_stips';
    % cates_raw = dir(cates_dir);
    cates_raw = ls(cates_dir);
    cates = cates_raw(3:end,:); 
    C = size(cates,1);   
    
    path_cline_train = ['../data/cline_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    path_cline_testa = ['../data/cline_testa_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    path_label_train = ['../data/label_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    path_label_testa = ['../data/label_train_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    path_stips_train = ['../data/stips_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    path_stips_testa = ['../data/stips_testa_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    path_centroids = ['../data/stips_centroids_r',int2str(round),'_f1', '_s', int2str(ns1), '_k', int2str(K), '.txt'];
       
    cline_train = importdata(path_cline_train); 
    cline_testa = importdata(path_cline_testa);
    label_train = importdata(path_label_train); 
    label_testa = importdata(path_label_testa); 
    stips_train = importdata(path_stips_train); 
    stips_testa = importdata(path_stips_testa); 
    centroids = importdata(path_centroids); 
end


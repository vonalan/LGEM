function [x_train, x_testa] = cluster(centroids)
    global cline_train stips_train cline_testa stips_testa;
    
    K = size(centroids,1);
    
    idx_train = knnsearch(centroids, stips_train);
    idx_testa = knnsearch(centroids, stips_testa);
    
    x_train = build_hist(cline_train, idx_train, K);
    x_testa = build_hist(cline_testa, idx_testa, K);
end 


function [bovfs]  = build_hist(cline, xdata, nbins)
    bovfs = zeros(size(cline, 1), nbins); 
    
    cbegin = 1; 
    for i = 1:size(cline, 1)
        cend = cbegin + cline(i,0); 
        hist = tabulate(xdata(cbegin:cend-1,:))'; 
        bovfs(i,hist(1,:)) = hist(2,:); 
        cbegin = cend;
    end;
end
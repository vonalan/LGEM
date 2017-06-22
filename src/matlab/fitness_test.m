global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;

round = 1; 
K = 128;
ns1 = 10000; 
ns2 = 2000;
[~] = load(round, K, ns1, ns2); 

mop = {}; 
mop.od = 2; 
mop.pd = 128; 
popsize = 100; 

mop.domain = [zeros(1,mop.pd);ones(1,mop.pd)];

randarray  = rand(popsize, mop.pd);
lowend     = mop.domain(1, :);
span       = mop.domain(2, :) - lowend;
parent_pop = round(randarray .* (span(ones(1, popsize), :)) + lowend(ones(1, popsize), :));

% global c_train y_train x_train c_testa y_testa x_testa;
O = fitness(parent_pop); 
fprintf('\n'); 
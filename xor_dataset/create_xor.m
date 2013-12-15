function create_xor(fname_out, n)
%creates dataset (Xtrain/Xtest/...) with xor problem; 
% n is number of datapoints per quadrant


Xtrain = normrnd(1, 0.1, 4*n, 2);
Xtrain(n+1:2*n, :) = - Xtrain(n+1:2*n, :);
Xtrain(2*n+1:3*n, 1) = - Xtrain(2*n+1:3*n, 1);
Xtrain(3*n+1:4*n, 2) = - Xtrain(3*n+1:4*n, 2);

Ytrain = ones(n*4, 1);
Ytrain(2*n+1:end) = -1;

Xtest = [];
Ytest = [];

save(fname_out, 'Xtrain', 'Xtest', 'Ytrain', 'Ytest');
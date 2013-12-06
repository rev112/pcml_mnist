function create_mini_xor(fname_out)

%creates dataset (Xtrain/Xtest/...) with xor problem; 
% n is number of datapoints per quadrant

Xtrain = [0, 0; 1, 1; 0, 1; 1, 0];
Ytrain = [1; 1; -1; -1];

Xtest = [];
Ytest = [];

save(fname_out, 'Xtrain', 'Xtest', 'Ytrain', 'Ytest');
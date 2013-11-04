function split_dataset(filename_in, filename_out)
% a function that splits files of MNIST dataset of two digits into training
% and validation sets


load(filename_in);

N = size(Xtrain, 1) / 2;

index1 = randperm(N);
index2 = randperm(N) + N;

i1_train = index1(1 : N*2/3);
i1_valid = index1(N*2/3 + 1 : end);
i2_train = index2(1 : N*2/3);
i2_valid = index2(N*2/3 + 1 : end);

TrainSet = Xtrain([i1_train i2_train], :);
ValidSet = Xtrain([i1_valid i2_valid], :);
TrainClass = Ytrain([i1_train i2_train], :);
ValidClass = Ytrain([i1_valid i2_valid], :);
TestSet = Xtest;
TestClass = Ytest;

save(filename_out, 'TrainSet', 'ValidSet', 'TrainClass', 'ValidClass', 'TestSet', 'TestClass');
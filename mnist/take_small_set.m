function take_small_set(filename_in, filename_out)
% a function that splits files of MNIST dataset of two digits into training
% and validation sets


load(filename_in);

N = size(TrainSet, 1) / 2;
VN = size(ValidSet, 1) / 2;

train_indices = [1:100, N+1:N+100];
valid_indices = [1:50, VN+1:VN+50];

TrainSet = TrainSet(train_indices, :);
TrainClass = TrainClass(train_indices, :);
ValidSet = ValidSet(valid_indices, :);
ValidClass = ValidClass(valid_indices, :);

save(filename_out, 'TrainSet', 'ValidSet', 'TrainClass', 'ValidClass', 'TestSet', 'TestClass');
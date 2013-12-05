function lower_resolution(filename_in, filename_out)
% a function that lowers resolution of images in MNIST dataset
% expected file is downloadable dataset, and not the split dataset


load(filename_in);

N = size(Xtrain, 1);
Xtrain_small = zeros(N, 12*12);

for i = 1:N
    img = lower(Xtrain(i, :));
    Xtrain_small(i, :) = img;
end

Xtrain = Xtrain_small;
clear Xtrain_small;

N = size(Xtest, 1);
Xtest_small = zeros(N, 12*12);

for i = 1:N
    img = lower(Xtest(i,:));
    Xtest_small(i,:) = img;
end

Xtest = Xtest_small;
clear Xtest_small;

save(filename_out, 'Xtrain', 'Xtest', 'Ytrain', 'Ytest');


function small = lower(big)
big = reshape(big, 28, 28);
small = imresize(big, [12, 12]);
small = reshape(small, 1, 12*12);
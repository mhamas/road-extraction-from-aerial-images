addpath('GraphCut')

prob = im2double(imread('../../results/CNN_output/test/high_res_raw/raw_test_4_pixels.png'));
I = im2double(imread('../../data/test_set/test_4.png'));

% Get the unaries
[h, w] = size(prob);

fgVal = -log(prob);
bgVal = -log(1 - prob);

unaries(:, 1) = fgVal(:);
unaries(:, 2) = bgVal(:);

% unaries = getUnaries(I, cost_fg, cost_bg, seed_fg, seed_bg); % complete getUnaries.m
im1 = reshape(unaries(:, 1), size(I, 1), size(I, 2));
im2 = reshape(unaries(:, 2), size(I, 1), size(I, 2));

% Get the pairwise
% Get lambda
lambda = 0.1
pairwise = getPairWise(I, lambda); % complete getPairWise.m

% TASK 2.4: Graph Cut

Handle = BK_Create(size(I, 1) * size(I, 2));
BK_SetUnary(Handle, fliplr(unaries)');
BK_SetNeighbors(Handle, pairwise);
BK_Minimize(Handle);
labels = BK_GetLabeling(Handle);
labels = labels - 1;

% Show the results
figure();
for i = 1:length(labels)
    if(labels(i))
        I(h*w+i)   = 0; % label 0: red
        I(2*h*w+i) = 0;
    else
        I(i)       = 0; % label 1: blue
        I(h*w+i)   = 0;
    end
end
subplot(2, 2, 1)
imshow(I);
title(['Lambda ' num2str(lambda)])

tmp = reshape(labels, size(prob));
output = imgToPrediction(tmp);

subplot(2, 2, 2)
imshow(imresize(output, 16, 'nearest'))



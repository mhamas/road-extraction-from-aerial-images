addpath('GraphCut')

% prob = im2double(imread('satImage_001_prob.png'));
prob = im2double(imread('raw_satImage_001_pixels.png'));
I = im2double(imread('satImage_001.png'));

% Get the unaries
[h, w] = size(prob);

fgVal = -log(prob);
bgVal = -log(1 - prob);

unaries(:, 1) = fgVal(:);
unaries(:, 2) = bgVal(:);

% unaries = getUnaries(I, cost_fg, cost_bg, seed_fg, seed_bg); % complete getUnaries.m
im1 = reshape(unaries(:, 1), size(I, 1), size(I, 2));
im2 = reshape(unaries(:, 2), size(I, 1), size(I, 2));
figure(2);
imagesc([im1 im2]);

% Get the pairwise
% Get lambda
lambda = 0
pairwise = getPairWise(I, lambda); % complete getPairWise.m

% TASK 2.4: Graph Cut

Handle = BK_Create(size(I, 1) * size(I, 2));
BK_SetUnary(Handle, fliplr(unaries)');
BK_SetNeighbors(Handle, pairwise);
BK_Minimize(Handle);
labels = BK_GetLabeling(Handle);
labels = labels - 1;

% labels = zeros(size(labels));
% labels(unaries(:, 1) >= unaries(:, 2)) = 1;
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

imshow(I);
title(['Lambda ' num2str(lambda)])

ptcSize = 16;

tmp = reshape(labels, size(prob));
output = zeros(25, 25);
idxI = 1;
for i = 1:ptcSize:h
    idxJ = 1;
    for j = 1:ptcSize:w
        if mean(mean(tmp(i:i+ptcSize-1, j:j+ptcSize-1))) > 0.25
            output(idxI, idxJ) = 1;
        end
        idxJ = idxJ + 1;
    end
    idxI = idxI + 1;
end

figure;
imshow(imresize(output, 16, 'nearest'))



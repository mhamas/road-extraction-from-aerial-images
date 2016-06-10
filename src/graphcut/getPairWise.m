function pairWise = getPairWise(I, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get pairwise terms for each pairs of pixels on image I and for
% regularizer lambda.
% 
% INPUT :
% - I      : color image
% - lambda : regularizer parameter
% 
% OUTPUT :
% - pairwise : sparse square matrix containing the pairwise costs for image
%              I and parameter lambda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Consider 8 neighborhood for each pixel

sigma = 5;
numRows = size(I, 1);
numCols = size(I, 2);
N = numRows * numCols;
% shifts =  [1 0; 1 1; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1];
shifts =  [1 0; 0 1; -1 0; 0 -1];
neighbors = size(shifts, 1);

i_indices = reshape(1:N, numRows, numCols); %indices of all image pixels
i_indices = i_indices(2:numRows - 1, 2:numCols - 1); % crop to not use borders for simplicity
i_indices = i_indices(:)';
idxI = repmat(i_indices, 1, neighbors); % replicate for each offset



offsets = [shifts(:, 1) + shifts(:, 2) * numRows]';
offsets = repmat(offsets, length(i_indices), 1);
offsets = offsets(:)';
idxJ = idxI + offsets; % J now contains neighbor indices for each idx I

s = zeros(1, neighbors * (numRows - 2) * (numCols - 2)); % s will hold all weights
k = 1;
for i = 1:neighbors
    s(k:k + length(i_indices) - 1) = smoothnessTerm(I, circshift(I, -shifts(i, :)), -shifts(i, :), lambda, sigma);
    k = k + length(i_indices);
end

idx = idxI < idxJ;
pairWise = sparse(idxI(idx), idxJ(idx), s(idx), N, N);

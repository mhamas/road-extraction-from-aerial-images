function unaries = getUnaries(I,hist_fg,hist_bg, seed_fg, seed_bg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get Unaries for all pixels in inputImg, using the foreground and
% background color histograms, and enforcing hard constraints on pixels
% marked by the user as foreground and background
% 
% INPUT :
% - I       : color image
% - hist_fg : foreground color histogram
% - hist_bg : background color histogram
% - seed_fg : pixels marked as foreground by the user
% - seed_bg : pixels marked as background by the user
% 
% OUTPUT :
% - unaries : Nx2 matrix containing the unary cost for every pixels in I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for each pixel, evaluate probabilites to belong to either fore- or
% background using histograms

% compute hist indices
hIdx = max(ceil(double(I) / double(255) * 32), 1);
hIdx = reshape(hIdx, [], 3);
hIdx = sub2ind([32 32 32], hIdx(:, 1), hIdx(:, 2), hIdx(:, 3));

fgVal = -log(hist_fg(hIdx));
bgVal = -log(hist_bg(hIdx));

R_fSum = reshape(fgVal, size(I, 1), size(I, 2));
R_bSum = reshape(bgVal, size(I, 1), size(I, 2));

% set unaries for dataterms
for i = 1:size(seed_fg, 1)
    R_fSum(seed_fg(i, 2), seed_fg(i, 1)) = 0;
    R_bSum(seed_fg(i, 2), seed_fg(i, 1)) = inf;
end
for i = 1:size(seed_bg, 1)
    R_fSum(seed_bg(i, 2), seed_bg(i, 1)) = inf;
    R_bSum(seed_bg(i, 2), seed_bg(i, 1)) = 0;
end

unaries(:, 1) = R_fSum(:);
unaries(:, 2) = R_bSum(:);



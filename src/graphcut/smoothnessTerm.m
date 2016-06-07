function [ result ] = smoothnessTerm( img, shifted_img, shift, lambda, sigma )
d = img - shifted_img;
result = lambda * exp(-sum(d .* d, 3) / 3 / (2 * sigma^2)) / (sqrt(sum(shift .^ 2)));
result = result(2:size(img, 1) - 1, 2:size(img, 2) - 1); % crop result
end


function [ output ] = imgToPrediction( img )
ptcSize = 16;

output = zeros(size(img, 1) / ptcSize, size(img, 2) / ptcSize);
idxI = 1;
for i = 1:ptcSize:size(img, 1)
    idxJ = 1;
    for j = 1:ptcSize:size(img, 2)
        if mean(mean(img(i:i+ptcSize-1, j:j+ptcSize-1))) > 0.25
            output(idxI, idxJ) = 1;
        end
        idxJ = idxJ + 1;
    end
    idxI = idxI + 1;
end

end


img = im2double(imread('../data/CNN_Output/Training/Probabilities/raw_satImage_007.png'));
ptcSize = 16;
img = imresize(img, size(img) / ptcSize, 'nearest');

h = size(img, 1);
w = size(img, 2);
Ix = zeros(h, w);
Ix2 = zeros(h, w);
p = zeros(h, w, 2); %dual variable

eps = 1e-40;
f = log(max(1 - img, eps)) - log(max(img, eps));
f = -f;
% f = min(1, max(0, f));

tau = 0.35;
sigma = 0.35;
lambda = 0.5;

theta = 1;
tic
for k = 1:2000
    
  fprintf('.');
  if(~mod(k,50))
      fprintf(sprintf(' (%d iterations)\n', k));
  end
  
  % Primal Dual iterations
  % update dual
  gradX = Ix2 - circshift(Ix2, [-1 0]);
  gradX(end, :) = 0;  
  gradY = Ix2 - circshift(Ix2, [0 -1]);
  gradY(:, end) = 0;  
  grad = cat(3, gradX, gradY);
  
  q = p + sigma * grad;
  p = bsxfun(@rdivide, q, max(1, sqrt(sum(q .^ 2, 3))));
  
  % update primal
  gradX = -p(:,:,1) + circshift(p(:,:,1), [1 0]);
  gradX(1, :) = 0;  
  gradY = -p(:,:,2) + circshift(p(:,:,2), [0 1]);
  gradY(:, 1) = 0;
  divP = gradX + gradY;

  tmp = Ix + tau * divP;
  Ix_new = min(1, max(0, tmp + lambda * tau * f));
  
  % third step
  Ix2 = Ix_new + theta * (Ix_new - Ix);
  Ix = Ix_new; 
end

figure(1);
subplot(1, 2, 1);
imshow(imresize(img, 10, 'nearest'));
subplot(1, 2, 2);
imshow(imresize(Ix2, 10, 'nearest'));

function [W1, W2, V1, V2, A, BT, C1, C2] ...
    = muscat_incremental_sparsa_time_postUpdate...
    (X1_T, X2_T, Y, W1_old, W2_old, V1_old, V2_old, ...
    A_old, C1_old, C2_old, lambda2, eta2, beta2, R)
% Incremental learning over space: add XS with size T x d

% Input:
% X1_T: S x d1
% X2_T: S x d2
% Y: S x 1
% BT_old should be a random vector.

% Output:
% A: S x R
% B: T x R
% C1: d1 x R
% C2: d2 x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

rng(0);
BT_old = rand(R,1);

[S, d1] = size(X1_T);
[S, d2] = size(X2_T);

vect = [A_old(:); BT_old(:); C1_old(:); C2_old(:); W1_old(:); W2_old(:); V1_old(:); V2_old(:)];
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, X1_T, X2_T, Y, ...
    lambda2, eta2, R,S, d1, d2, W1_old, W2_old, V1_old, V2_old, A_old, BT_old, C1_old, C2_old);
% non-negativen l1 norm proximal operator.
non_smooth = prox_L1(beta2);
sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 20  ,...
        'maxfunEv'      , 50  ,... % max number of function evaluations
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[vect_result, ~,info] = pnopt_sparsa( smoothF, non_smooth, vect, sparsa_options );

if isnan(sum(vect_result))
    A = A_old; BT = BT_old; C1 = C1_old; C2 = C2_old; W1 = W1_old; W2 = W2_old; V1 = V1_old; V2 = V2_old;
else
%     A = reshape(vect_result(1 : length(A_old(:))), size(A_old));
%     BT = reshape(vect_result(length(A_old(:)) + 1 : length(A_old(:)) + length(BT_old(:))), size(BT_old));
%     C1 = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ 1 : length(A_old(:)) + length(BT_old(:)) + length(C1_old(:))), size(C1_old));
%     W1 = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ length(C1_old(:)) + 1 : length(A_old(:)) + length(BT_old(:)) + length(C1_old(:)) + length(W1_old(:))), size(W1_old));
%     V1 = reshape(vect_result(length(A_old(:)) + length(BT_old(:))+ length(C1_old(:)) + length(W1_old(:)) + 1 : length(A_old(:)) + length(BT_old(:)) + length(C1_old(:)) + length(W1_old(:)) + length(V1_old(:))), size(V1_old));

    currentPos = 0; nextPos = currentPos + length(A_old(:)); shapeSize = size(A_old);
    A = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(BT_old(:)); shapeSize = size(BT_old);
    BT = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(C1_old(:)); shapeSize = size(C1_old);
    C1 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(C2_old(:)); shapeSize = size(C2_old);
    C2 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(W1_old(:)); shapeSize = size(W1_old);
    W1 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(W2_old(:)); shapeSize = size(W2_old);
    W2 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(V1_old(:)); shapeSize = size(V1_old);
    V1 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(V2_old(:)); shapeSize = size(V2_old);
    V2 = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
end
end
function [f, g] = smooth_part(parameterVect, X1_T, X2_T, Y, ...
    lambda2, eta2, R, S, d1, d2, W1_old, W2_old, V1_old, V2_old, A_old, BT_old, C1_old, C2_old)
% recover the models from the last iteration

% A = reshape(parameterVect(1 : S*R), [S, R]);
% BT = reshape(parameterVect(length(A(:)) + 1 : length(A(:)) + R), [R, 1]);
% C = reshape(parameterVect(length(A(:)) + length(BT(:))+ 1 : length(A(:)) + length(BT(:)) + d1*R), [d1, R]);
% W = reshape(parameterVect(length(A(:)) + length(BT(:))+ length(C(:)) + 1 : length(A(:)) + length(BT(:)) + length(C(:)) + R * d1), [R, d1]);
% V = reshape(parameterVect(length(A(:)) + length(BT(:))+ length(C(:)) + length(W(:)) + 1 : length(A(:)) + length(BT(:)) + length(C(:)) + length(W(:)) + R * d1), [R, d1]);

currentPos = 0; nextPos = currentPos + length(A_old(:)); shapeSize = size(A_old);
A = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(BT_old(:)); shapeSize = size(BT_old);
BT = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(C1_old(:)); shapeSize = size(C1_old);
C1 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(C2_old(:)); shapeSize = size(C2_old);
C2 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(W1_old(:)); shapeSize = size(W1_old);
W1 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(W2_old(:)); shapeSize = size(W2_old);
W2 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(V1_old(:)); shapeSize = size(V1_old);
V1 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(V2_old(:)); shapeSize = size(V2_old);
V2 = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);

% compute f
% ==========================================================
% loss fast
% loss = 0.5 * norm(sum(XS .* bsxfun(@plus, AS' * W, B'*V),2) - Y)^2;
sloss = sum(X1_T .* bsxfun(@plus, A * W1, BT'*V1),2) + sum(X2_T .* bsxfun(@plus, A * W2, BT'*V2),2) - Y; 
loss = 0.5 * norm(sloss)^2;

X1_T_hat = double(ktensor({A, BT', C1}));
X2_T_hat = double(ktensor({A, BT', C2}));

X1TT = tensor(reshape(X1_T, [S, 1, d1]));
X2TT = tensor(reshape(X2_T, [S, 1, d2]));

regularizer = 0.5 * lambda2 * norm(X1TT(:) - X1_T_hat(:))^2 + ...
    0.5 * lambda2 * norm(X2TT(:) - X2_T_hat(:))^2 + ...
    0.5 * eta2 * (norm(W1-W1_old, 'fro')^2 + norm(W2-W2_old, 'fro')^2 +...
    norm(V1 - V1_old, 'fro')^2 + norm(V2 - V2_old, 'fro')^2 +...
    norm(A - A_old, 'fro')^2 + norm(C1 - C1_old, 'fro')^2 + norm(C2 - C2_old, 'fro')^2 );
f = loss + regularizer;


% compute gradient
% ======================================================
% A
C1KB = kr(C1,BT');
C2KB = kr(C2,BT');
X1M1 = double(tenmat(X1TT, 1));
X2M1 = double(tenmat(X2TT, 1));

g_A = bsxfun(@times, sloss, (X1_T*W1' + X2_T*W2')) - ...
    lambda2 * (X1M1 - A * C1KB')*C1KB - lambda2 * (X2M1 - A * C2KB')*C2KB +...
    eta2 * (A - A_old); 

% BT
X1M2 = double(tenmat(X1TT, 2));
X2M2 = double(tenmat(X2TT, 2));

C1KA = kr(C1, A);
C2KA = kr(C2, A);
g_BT = (V1 * X1_T' + V2 * X2_T') * sloss - ...
    lambda2 * ((X1M2 - BT' * C1KA') * C1KA)' - lambda2 * ((X2M2 - BT' * C2KA') * C2KA)';

% C1
BKA = kr(BT', A);
X1M3 = double(tenmat(X1TT, 3));
g_C1 = -lambda2 * (X1M3 - C1 * BKA') * BKA + eta2 * (C1 - C1_old);

% C2
X2M3 = double(tenmat(X2TT, 3));
g_C2 = -lambda2 * (X2M3 - C2 * BKA') * BKA + eta2 * (C2 - C2_old);
% ==================================================================
% W1
g_W1 = A' * bsxfun(@times, X1_T, sloss) + eta2 * (W1 - W1_old);
% W
g_W2 = A' * bsxfun(@times, X2_T, sloss) + eta2 * (W2 - W2_old);
% V1
g_V1 = BT * sum(bsxfun(@times, X1_T, sloss), 1) + eta2 * (V1 - V1_old);
% V2
g_V2 = BT * sum(bsxfun(@times, X2_T, sloss), 1) + eta2 * (V2 - V2_old);
% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_A = zeros(S * R, 1); % good
% g_BT = zeros(R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_A(:); g_BT(:); g_C1(:); g_C2(:); g_W1(:); g_W2(:); g_V1(:); g_V2(:)];

end




function op = prox_L1(beta2) 

%PROX_L1    L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
% Dual: proj_linf.m

% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes
%  lasso for VT
    function v = f(x)
        v = beta2 * norm(x,1);
    end

    function x = prox_f(x,t)
        tq = t .* beta2; % March 2012, allowing vectorized stepsizes
        s  = 1 - min( tq./abs(x), 1 );
        x  = x .* s;
    end
end

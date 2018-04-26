function [W1, W2, V1, V2, AS, B, C1, C2] ...
    = muscat_incremental_sparsa_space...
    (X1_S, X2_S, Y, W1_old, W2_old, V1_old, V2_old, ...
    B_old, C1_old, C2_old, lambda1, eta1, beta1, R)
% Incremental learning over space: add X1_S and X2_S with size T x d1/d2

% Input:
% X1_S: T x d1
% X2_S: T x d2
% Y: T x 1
% As_old should be a random vector.

% Output:
% A: S x R
% B: T x R
% C1: d x R
% C2: d x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

rng(0);

AS_old = rand(R,1);

[T, d1] = size(X1_S);
[T, d2] = size(X2_S);

vect = [AS_old(:); B_old(:); C1_old(:); C2_old(:); W1_old(:); W2_old(:); V1_old(:); V2_old(:)];
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, X1_S, X2_S, Y, ...
    lambda1, eta1, R,T, d1, d2, W1_old, W2_old, V1_old, V2_old, AS_old, B_old, C1_old, C2_old);
% non-negativen l1 norm proximal operator.
non_smooth = prox_L1(beta1);
sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 20  ,...
    'maxfunEv'  , 50  ,... % max number of function evaluations
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[vect_result, ~,info] = pnopt_sparsa( smoothF, non_smooth, vect, sparsa_options );

if isnan(sum(vect_result))
    AS = AS_old; B = B_old; C1 = C1_old; C2 = C2_old; W1 = W1_old; W2 = W2_old; V1 = V1_old; V2 = V2_old;
else
    currentPos = 0; nextPos = currentPos + length(AS_old(:)); shapeSize = size(AS_old);
    AS = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
    currentPos = nextPos; nextPos = currentPos + length(B_old(:)); shapeSize = size(B_old);
    B = reshape(vect_result(currentPos + 1 : nextPos), shapeSize);
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
function [f, g] = smooth_part(parameterVect, X1_S, X2_S, Y, ...
    lambda1, eta1, R, T, d1, d2, W1_old, W2_old, V1_old, V2_old, AS_old, B_old, C1_old, C2_old)
% recover the models from the last iteration

% AS = parameterVect(1 : R);
% B = reshape(parameterVect(length(AS(:)) + 1 : length(AS(:)) + T*R), [T, R]);
% C = reshape(parameterVect(length(AS(:)) + length(B(:))+ 1 : length(AS(:)) + length(B(:)) + d1*R), [d1, R]);
% W = reshape(parameterVect(length(AS(:)) + length(B(:))+ length(C(:)) + 1 : length(AS(:)) + length(B(:)) + length(C(:)) + R * d1), [R, d1]);
% V = reshape(parameterVect(length(AS(:)) + length(B(:))+ length(C(:)) + length(W(:)) + 1 : length(AS(:)) + length(B(:)) + length(C(:)) + length(W(:)) + R * d1), [R, d1]);


currentPos = 0; nextPos = currentPos + length(AS_old(:)); shapeSize = size(AS_old);
AS = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
currentPos = nextPos; nextPos = currentPos + length(B_old(:)); shapeSize = size(B_old);
B = reshape(parameterVect(currentPos + 1 : nextPos), shapeSize);
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
tloss = sum(X1_S .* bsxfun(@plus, AS' * W1, B*V1),2) + sum(X2_S .* bsxfun(@plus, AS' * W2, B*V2),2) - Y;
loss = 0.5 * norm(tloss)^2;

% loss slow
% loss = 0;
% for t = 1:T
%     loss = loss + 0.5 * norm(XS(t,:) * (W' * AS + V' * B(t,:)') - Y(t))^2;
% end

X1_T_hat = double(ktensor({AS', B, C1}));
X2_T_hat = double(ktensor({AS', B, C2}));
% XT_hat = squeeze(double(XT_hat));
X1_ST = tensor(reshape(X1_S, [1 T d1]));
X2_ST = tensor(reshape(X2_S, [1 T d2]));

regularizer = 0.5 * lambda1 * norm(X1_ST(:) - X1_T_hat(:))^2 + ...
    0.5 * lambda1 * norm(X2_ST(:) - X2_T_hat(:))^2 + ...
    0.5 * eta1 * (norm(W1-W1_old, 'fro')^2 + norm(W2-W2_old, 'fro')^2 + ...
    norm(V1 - V1_old, 'fro')^2 + norm(V2 - V2_old, 'fro')^2 +...
    norm(B - B_old, 'fro')^2 + norm(C1 - C1_old, 'fro')^2 + norm(C2 - C2_old, 'fro')^2 );
f = loss + regularizer;


% compute gradient
% ======================================================
% A(S+1)
C1KB = kr(C1,B);
C2KB = kr(C2,B);
X1M1 = double(tenmat(X1_ST, 1));
X2M1 = double(tenmat(X2_ST, 1));

g_AS = (W1 * X1_S' + W2 * X2_S') * tloss ...
    - lambda1 * ((X1M1 - AS' * C1KB')*C1KB)' - lambda1 * ((X2M1 - AS' * C2KB')*C2KB)';
% g_AS = W * XS' * tloss;

% B
X1M2 = double(tenmat(X1_ST, 2));
X2M2 = double(tenmat(X2_ST, 2));

C1KA = kr(C1, AS');
C2KA = kr(C2, AS');
g_B = bsxfun(@times, tloss, (X1_S * V1' + X2_S * V2')) ...
    - lambda1 * ((X1M2 - B * C1KA') * C1KA) - lambda1 * ((X2M2 - B * C2KA') * C2KA)...
    + eta1 * (B - B_old);

% C1
BKA = kr(B, AS');
X1M3 = double(tenmat(X1_ST, 3));
g_C1 = -lambda1 * (X1M3 - C1 * BKA') * BKA + eta1 * (C1 - C1_old);

% C2
X2M3 = double(tenmat(X2_ST, 3));
g_C2 = -lambda1 * (X2M3 - C2 * BKA') * BKA + eta1 * (C2 - C2_old);

% ==================================================================
% W1
g_W1 = AS * sum(bsxfun(@times, X1_S, tloss), 1) + eta1 * (W1 - W1_old);
% W2
g_W2 = AS * sum(bsxfun(@times, X2_S, tloss), 1) + eta1 * (W2 - W2_old);

% V1
g_V1 = B' * bsxfun(@times, X1_S, tloss) + eta1 * (V1 - V1_old);
% V2
g_V2 = B' * bsxfun(@times, X2_S, tloss) + eta1 * (V2 - V2_old);

% ==================================================================
% Set the gradients to be non-zero if we want to test that gradient
% g_AS = zeros(R, 1); % good
% g_B = zeros(T * R, 1); % good
% g_C = zeros(d * R, 1); % good
% g_W = zeros(R * d, 1); % uncomment to check correctness
% g_V = zeros(R * d, 1); % uncomment to check correctness


g = [g_AS(:); g_B(:); g_C1(:); g_C2(:); g_W1(:); g_W2(:); g_V1(:); g_V2(:)];

end




function op = prox_L1(beta1)

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
        v = beta1 * norm(x,1);
    end

    function x = prox_f(x,t)
        tq = t .* beta1; % March 2012, allowing vectorized stepsizes
        s  = 1 - min( tq./abs(x), 1 );
        x  = x .* s;
    end
end

function [BT] = muscat_incremental_sparsa_time_preUpdate(X1_T, X2_T, A_old, C1_old, C2_old, R, lambda2, beta2)
% Incremental learning over space: add XS with size T x d

% Input:
% X1_T: S x d1
% X2_T: S x d2

% BT_old should be a random vector.

% A: S x R
% B: T x R
% C1: d x R
% C2: d x R
% use tensor toolbox
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met/');

rng(0);
BT_old = rand(R,1);

[S, d1] = size(X1_T);
[S, d2] = size(X2_T);

vect = BT_old;
%
% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part(parameterVect, X1_T, X2_T, lambda2, ...
    R,S, d1, d2, A_old, C1_old, C2_old);
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
    BT = BT_old;
else
    BT = vect_result;
end
end

function [f, g] = smooth_part(parameterVect, X1_T, X2_T, lambda2, ...
    R, S, d1, d2, A_old, C1_old, C2_old)
% recover the models from the last iteration

BT = parameterVect;
% compute f
% ==========================================================
loss = 0;

X1_T_hat = double(ktensor({A_old, BT', C1_old}));
X1_TT = tensor(reshape(X1_T, [S 1 d1]));

X2_T_hat = double(ktensor({A_old, BT', C2_old}));
X2_TT = tensor(reshape(X2_T, [S 1 d2]));

regularizer = 0.5 * lambda2 * (norm(X1_TT(:) - X1_T_hat(:))^2 ...
    + norm(X2_TT(:) - X2_T_hat(:))^2);
f = loss + regularizer;

% compute gradient
% ======================================================
X1M = double(tenmat(X1_TT, 2));
X2M = double(tenmat(X2_TT, 2));
CKA_1 = kr(C1_old, A_old);
CKA_2 = kr(C2_old, A_old);
g_BT = - lambda2 * ((X1M - BT' * CKA_1') * CKA_1)'  ...
    - lambda2 * ((X2M - BT' * CKA_2') * CKA_2)';

g = g_BT;

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

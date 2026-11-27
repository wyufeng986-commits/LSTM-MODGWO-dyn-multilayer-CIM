function P = calculatePropagationProbability(W, beta, epsVal)
%CALCULATEPROPAGATIONPROBABILITY  Weight-driven IC edge probabilities.
%   P(i,j) = 1 - exp(-beta * w_tilde(i,j)) for existing arcs.
%   w_tilde = W / (max(W(:)) + eps), original W is preserved by caller.
%
%   Convention: W(i,j) > 0 means i can influence j.

if nargin < 2 || isempty(beta)
    beta = 1.0;
end
if nargin < 3 || isempty(epsVal)
    epsVal = 1e-12;
end
if ~issparse(W)
    W = sparse(W);
end
if nnz(W) == 0
    P = sparse(size(W, 1), size(W, 2));
    return;
end

wmax = full(max(max(W)));
if ~(isfinite(wmax) && wmax > 0)
    wmax = 1;
end
Wtil = W ./ (wmax + epsVal);
[ii, jj, vv] = find(Wtil);
pp = 1 - exp(-beta .* vv);
pp = min(max(pp, 0), 1);
P = sparse(ii, jj, pp, size(W, 1), size(W, 2));
end

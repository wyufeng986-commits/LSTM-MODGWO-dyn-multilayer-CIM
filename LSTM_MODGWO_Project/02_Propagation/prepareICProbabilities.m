function cache = prepareICProbabilities(Network, opts)
%PREPAREICPROBABILITIES  Precompute IC edge probabilities for all snapshots/layers.

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'beta') || isempty(opts.beta), opts.beta = 1.0; end
if ~isfield(opts, 'epsVal') || isempty(opts.epsVal), opts.epsVal = 1e-12; end
if ~isfield(opts, 'includeFacebook'), opts.includeFacebook = false; end

layerIdx = selectPropagationLayers(Network, opts);
Luse = numel(layerIdx);
T = Network.numSnapshots;
N = Network.numNodes;
beta = opts.beta;
if isscalar(beta)
    betaVec = repmat(double(beta), 1, Network.numLayers);
else
    betaVec = double(beta(:))';
end

Pcell = cell(Luse, T);
log1mP = cell(Luse, T);
for li = 1:Luse
    l = layerIdx(li);
    b = betaVec(l);
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W); W = sparse(W); end
        P = calculatePropagationProbability(W, b, opts.epsVal);
        Pcell{li, t} = P;
        if nnz(P) == 0
            log1mP{li, t} = sparse(N, N);
        else
            [ii, jj, vv] = find(P);
            vv = log(max(1 - vv, realmin('double')));
            log1mP{li, t} = sparse(ii, jj, vv, N, N);
        end
    end
end

cache = struct();
cache.layerIdx = layerIdx;
cache.Pcell = Pcell;
cache.log1mP = log1mP;
cache.beta = betaVec(layerIdx);
cache.layerNames = string({Network.layer(layerIdx).name});
end

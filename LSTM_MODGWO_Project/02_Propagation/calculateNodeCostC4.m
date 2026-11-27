function costInfo = calculateNodeCostC4(Network, opts)
%CALCULATENODECOSTC4  Hybrid deployment cost factors (C4).
%   Cost(v) = alpha*sHat(v) + beta*LHat(v) + gamma*DHat(v)
%   Default alpha=beta=gamma=1/3 after max-normalization to [0,1].
%
%   sOut : sum of weighted out-strength over main dynamic layers & snapshots
%   L    : number of layers where v has at least one outgoing arc
%   D    : number of days where v has outgoing arc in any used layer

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'includeFacebook'), opts.includeFacebook = false; end
if ~isfield(opts, 'alpha') || isempty(opts.alpha), opts.alpha = 1/3; end
if ~isfield(opts, 'beta') || isempty(opts.beta), opts.beta = 1/3; end
if ~isfield(opts, 'gamma') || isempty(opts.gamma), opts.gamma = 1/3; end
if ~isfield(opts, 'layerNames'), opts.layerNames = []; end

N = Network.numNodes;
T = Network.numSnapshots;
layerIdx = selectPropagationLayers(Network, opts);

sOut = zeros(N, 1);
layerHit = false(N, numel(layerIdx));
dayHit = false(N, T);

for li = 1:numel(layerIdx)
    l = layerIdx(li);
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W); W = sparse(W); end
        outW = full(sum(W, 2));
        sOut = sOut + outW;
        hasOut = outW > 0;
        layerHit(:, li) = layerHit(:, li) | hasOut;
        dayHit(:, t) = dayHit(:, t) | hasOut;
    end
end

Lcount = sum(layerHit, 2);
Dcount = sum(dayHit, 2);

sHat = unit_max_normalize(sOut);
LHat = unit_max_normalize(Lcount);
DHat = unit_max_normalize(Dcount);

nodeCost = opts.alpha * sHat + opts.beta * LHat + opts.gamma * DHat;

costInfo = struct();
costInfo.nodeCost = nodeCost;
costInfo.sOut = sOut;
costInfo.Lcount = Lcount;
costInfo.Dcount = Dcount;
costInfo.sHat = sHat;
costInfo.LHat = LHat;
costInfo.DHat = DHat;
costInfo.alpha = opts.alpha;
costInfo.beta = opts.beta;
costInfo.gamma = opts.gamma;
costInfo.layerIdx = layerIdx;
costInfo.layerNames = string({Network.layer(layerIdx).name});
end

function y = unit_max_normalize(x)
x = x(:);
mx = max(x);
if ~(isfinite(mx) && mx > 0)
    y = zeros(size(x));
else
    y = x ./ mx;
end
end

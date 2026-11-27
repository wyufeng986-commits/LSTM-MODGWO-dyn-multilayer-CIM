function zS = extractSeedFeatures(Network, S_A, S_B, costInfo, opts)
%EXTRACTSEEDFEATURES  Static seed-set features for side A (and vs B).

if nargin < 5, opts = struct(); end
if ~isfield(opts, 'includeFacebook'), opts.includeFacebook = false; end

S_A = unique(S_A(:))';
S_B = unique(S_B(:))';
N = Network.numNodes;
layerIdx = selectPropagationLayers(Network, opts);
T = Network.numSnapshots;
L = numel(layerIdx);

% per-layer out/in strength of seeds aggregated over time
outA = zeros(1, L); inA = zeros(1, L);
outB = zeros(1, L);
for li = 1:L
    l = layerIdx(li);
    outSum = zeros(N, 1); inSum = zeros(N, 1);
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W), W = sparse(W); end
        outSum = outSum + full(sum(W, 2));
        inSum = inSum + full(sum(W, 1))';
    end
    outA(li) = sum(outSum(S_A));
    inA(li) = sum(inSum(S_A));
    outB(li) = sum(outSum(S_B));
end

% layer distribution entropy of A out-strength
p = outA / max(sum(outA), eps);
p = p(p > 0);
if isempty(p)
    ent = 0;
else
    ent = -sum(p .* log(p + eps));
end

% crude pairwise distance proxy: inverse co-neighbor overlap on union graph
Wuni = sparse(N, N);
for li = 1:L
    l = layerIdx(li);
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W), W = sparse(W); end
        Wuni = Wuni + spones(W);
    end
end
Wuni = spones(Wuni);
meanDistA = mean_seed_hop_proxy(Wuni, S_A);
meanDistAB = mean_cross_hop_proxy(Wuni, S_A, S_B);

% neighborhood overlap among A seeds
overlapA = seed_neighborhood_overlap(Wuni, S_A);

costA = sum(costInfo.nodeCost(S_A));

zS = [ ...
    numel(S_A); numel(S_B); ...
    sum(outA); sum(inA); sum(outB); ...
    mean(outA); mean(inA); ...
    ent; ...
    meanDistA; meanDistAB; overlapA; ...
    costA ...
    ];
zS = zS(:);
end

function d = mean_seed_hop_proxy(W, S)
if numel(S) < 2
    d = 0; return;
end
vals = [];
for i = 1:numel(S)
    for j = i+1:numel(S)
        vals(end+1) = hop_proxy(W, S(i), S(j)); %#ok<AGROW>
    end
end
d = mean(vals);
end

function d = mean_cross_hop_proxy(W, SA, SB)
vals = [];
for i = 1:numel(SA)
    for j = 1:numel(SB)
        vals(end+1) = hop_proxy(W, SA(i), SB(j)); %#ok<AGROW>
    end
end
if isempty(vals), d = 0; else, d = mean(vals); end
end

function d = hop_proxy(W, u, v)
% 1 if direct arc either way, else 2 if common neighbor, else 3
if W(u, v) > 0 || W(v, u) > 0
    d = 1; return;
end
nu = find(W(u, :) | W(:, u)');
nv = find(W(v, :) | W(:, v)');
if ~isempty(intersect(nu, nv))
    d = 2;
else
    d = 3;
end
end

function o = seed_neighborhood_overlap(W, S)
if numel(S) < 2
    o = 0; return;
end
acc = 0; cnt = 0;
for i = 1:numel(S)
    ni = find(W(S(i), :) | W(:, S(i))');
    for j = i+1:numel(S)
        nj = find(W(S(j), :) | W(:, S(j))');
        uni = numel(union(ni, nj));
        if uni > 0
            acc = acc + numel(intersect(ni, nj)) / uni;
            cnt = cnt + 1;
        end
    end
end
if cnt == 0, o = 0; else, o = acc / cnt; end
end

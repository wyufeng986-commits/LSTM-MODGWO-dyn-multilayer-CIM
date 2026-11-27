function Snew = discreteUpdate(S, leaders, potential, S_B, N, a)
%DISCRETEUPDATE  Three-leader discrete exchange + majority vote.
%   leaders: cell of wolves/structs with .Position (alpha,beta,delta)
%   a in [0,2]: GWO schedule controlling move intensity

S = unique(S(:))';
k = numel(S);
S_B = unique(S_B(:))';
if nargin < 6 || isempty(a), a = 1; end

nLead = numel(leaders);
props = cell(nLead, 1);
nMove = max(1, min(k, ceil(abs(a) * k / 2)));
for i = 1:nLead
    L = unique(leaders{i}.Position(:)');
    props{i} = move_toward(S, L, potential, S_B, N, nMove, k);
end

votes = containers.Map('KeyType', 'double', 'ValueType', 'double');
allSets = [{S}, props(:)'];
for s = 1:numel(allSets)
    u = unique(allSets{s});
    for j = 1:numel(u)
        v = u(j);
        if isKey(votes, v)
            votes(v) = votes(v) + 1;
        else
            votes(v) = 1;
        end
    end
end

keys = cell2mat(votes.keys);
vals = cell2mat(votes.values);
score = vals(:)' + 1e-3 * potential(keys);
score(~isfinite(score)) = -1e12;
[~, ord] = sort(score, 'descend');
ranked = keys(ord);
Snew = repairSeedSet(ranked(1:min(k, numel(ranked))), k, N, S_B, potential);

if rand < 0.25
    pool = setdiff(1:N, [Snew, S_B]);
    if ~isempty(pool) && k >= 1
        drop = Snew(randi(k));
        add = pool(weighted_index(potential(pool)));
        Snew = repairSeedSet([setdiff(Snew, drop, 'stable'), add], k, N, S_B, potential);
    end
end
end

function S2 = move_toward(S, L, potential, S_B, N, nMove, k)
onlyS = setdiff(S, L, 'stable');
onlyL = setdiff(L, [S, S_B], 'stable');
pool = setdiff(1:N, [S, S_B], 'stable');
if isempty(onlyL)
    onlyL = pool;
end
S2 = S;
nMove = min([nMove, numel(onlyS), numel(onlyL)]);
if nMove <= 0
    S2 = repairSeedSet(S2, k, N, S_B, potential);
    return;
end
[~, dOrd] = sort(potential(onlyS), 'ascend');
dropNodes = onlyS(dOrd(1:nMove));
addNodes = sample_nodes(onlyL, potential(onlyL), nMove);
S2 = setdiff(S2, dropNodes, 'stable');
S2 = [S2, addNodes];
S2 = repairSeedSet(S2, k, N, S_B, potential);
end

function nodes = sample_nodes(nodeList, w, m)
nodeList = nodeList(:)';
w = w(:)';
m = min(m, numel(nodeList));
nodes = zeros(1, m);
avail = true(size(nodeList));
for t = 1:m
    ww = w;
    ww(~avail) = -inf;
    j = weighted_index(ww);
    nodes(t) = nodeList(j);
    avail(j) = false;
end
end

function idx = weighted_index(w)
w = w(:)';
finite = isfinite(w);
if ~any(finite)
    idx = 1; return;
end
ww = w;
ww(~finite) = min(w(finite));
ww = ww - min(ww) + eps;
cdf = cumsum(ww / sum(ww));
idx = find(rand <= cdf, 1, 'first');
if isempty(idx), idx = numel(ww); end
end

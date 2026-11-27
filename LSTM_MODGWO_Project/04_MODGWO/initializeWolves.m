function wolves = initializeWolves(N, k, S_B, potential, popSize, rngSeed)
%INITIALIZEWOLVES  Mixed potential / random / diversity discrete init.

if nargin < 6 || isempty(rngSeed), rngSeed = 1; end
rng(rngSeed);

S_B = unique(S_B(:))';
pool = setdiff(1:N, S_B);
if numel(pool) < k
    error('initializeWolves:Pool', 'Feasible pool smaller than k.');
end

[~, potOrd] = sort(potential(pool), 'descend');
topPool = pool(potOrd(1:min(numel(pool), max(3 * k, 20))));

nPot = max(1, round(0.4 * popSize));
nDiv = max(1, round(0.2 * popSize));
nRnd = popSize - nPot - nDiv;
if nRnd < 0
    nPot = max(1, popSize - 2);
    nDiv = 1;
    nRnd = popSize - nPot - nDiv;
end

wolves = repmat(empty_wolf(), popSize, 1);
idx = 0;

for i = 1:nPot
    idx = idx + 1;
    pick = sample_biased(topPool, potential, k);
    wolves(idx) = make_wolf(repairSeedSet(pick, k, N, S_B, potential));
end

for i = 1:nRnd
    idx = idx + 1;
    pick = pool(randperm(numel(pool), k));
    wolves(idx) = make_wolf(repairSeedSet(pick, k, N, S_B, potential));
end

for i = 1:nDiv
    idx = idx + 1;
    pick = diversity_pick(pool, potential, k, wolves(1:idx));
    wolves(idx) = make_wolf(repairSeedSet(pick, k, N, S_B, potential));
end

% Deduplicate identical seeds by random repair
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:popSize
    key = mat2str(wolves(i).Position);
    guard = 0;
    while isKey(seen, key) && guard < 20
        guard = guard + 1;
        flip = pool(randi(numel(pool)));
        drop = wolves(i).Position(randi(k));
        trial = setdiff([wolves(i).Position, flip], drop, 'stable');
        wolves(i) = make_wolf(repairSeedSet(trial, k, N, S_B, potential));
        key = mat2str(wolves(i).Position);
    end
    seen(key) = true;
end
end

function w = empty_wolf()
w = struct('Position', [], 'SeedSet', [], 'Cost', [], ...
    'F_A', NaN, 'F_B', NaN, 'CostRaw', NaN, ...
    'evalMode', '', 'GridIndex', [], 'GridSubIndex', [], ...
    'detail', struct());
end

function w = make_wolf(S)
w = empty_wolf();
w.Position = S(:)';
w.SeedSet = w.Position;
end

function pick = sample_biased(pool, potential, k)
p = potential(pool);
p(~isfinite(p)) = min(p(isfinite(p)));
p = p - min(p) + eps;
p = p / sum(p);
pick = zeros(1, k);
avail = pool;
pav = p;
for t = 1:k
    cdf = cumsum(pav / sum(pav));
    j = find(rand <= cdf, 1, 'first');
    if isempty(j), j = 1; end
    pick(t) = avail(j);
    avail(j) = [];
    pav(j) = [];
end
end

function pick = diversity_pick(pool, potential, k, existing)
% Greedy: repeatedly add node maximizing min Jaccard distance to existing sets.
pick = [];
remain = pool;
if isempty(existing)
    pick = sample_biased(pool, potential, k);
    return;
end
sets = {existing.Position};
for t = 1:k
    bestScore = -inf; bestNode = remain(1);
    for j = 1:numel(remain)
        trial = [pick, remain(j)];
        dmin = inf;
        for s = 1:numel(sets)
            dmin = min(dmin, jaccard_dist(trial, sets{s}));
        end
        score = dmin + 1e-3 * potential(remain(j));
        if score > bestScore
            bestScore = score;
            bestNode = remain(j);
        end
    end
    pick(end + 1) = bestNode; %#ok<AGROW>
    remain = setdiff(remain, bestNode, 'stable');
end
end

function d = jaccard_dist(a, b)
a = unique(a); b = unique(b);
inter = numel(intersect(a, b));
uni = numel(union(a, b));
if uni == 0
    d = 0;
else
    d = 1 - inter / uni;
end
end

function wolf = localSearch(wolf, ctx, nTries)
%LOCALSEARCH  Single-node swap local polish; keep if not dominated.
if nargin < 3 || isempty(nTries), nTries = 3; end

N = ctx.N;
k = numel(wolf.Position);
S_B = ctx.S_B;
potential = ctx.potential;
pool = setdiff(1:N, [wolf.Position, S_B]);
if isempty(pool) || k < 1
    return;
end

best = wolf;
for t = 1:nTries
    drop = best.Position(randi(k));
    % Prefer low-cost high-potential additions
    add = pool(weighted_index(potential(pool)));
    Strial = repairSeedSet([setdiff(best.Position, drop, 'stable'), add], ...
        k, N, S_B, potential);
    trial = best;
    trial.Position = Strial;
    trial.SeedSet = Strial;
    trial = evaluateCandidate(trial, ctx);
    if better_or_nondominated(trial, best)
        best = trial;
        pool = setdiff(1:N, [best.Position, S_B]);
        if isempty(pool), break; end
    end
end
wolf = best;
end

function tf = better_or_nondominated(a, b)
% Accept if a dominates b, or mutually non-dominated with lower cost, or higher FA
if dominates(a.Cost, b.Cost)
    tf = true;
elseif dominates(b.Cost, a.Cost)
    tf = false;
else
    % tie-break: prefer higher FA then lower cost
    if a.F_A > b.F_A + 1e-9
        tf = true;
    elseif abs(a.F_A - b.F_A) <= 1e-9 && a.CostRaw < b.CostRaw
        tf = true;
    else
        tf = false;
    end
end
end

function dom = dominates(x, y)
x = x(:); y = y(:);
dom = all(x <= y) && any(x < y);
end

function idx = weighted_index(w)
w = w(:)';
finite = isfinite(w);
if ~any(finite), idx = 1; return; end
ww = w; ww(~finite) = min(w(finite));
ww = ww - min(ww) + eps;
cdf = cumsum(ww / sum(ww));
idx = find(rand <= cdf, 1, 'first');
if isempty(idx), idx = numel(ww); end
end

function result = LSTM_MODGWO(ctx, opts)
%LSTM_MODGWO  Multi-objective discrete GWO with optional LSTM surrogate.
%
% Objectives: max F_A, min Cost_C4
% Encoding: fixed-k unique S_A, disjoint from fixed S_B
%
% opts fields (defaults for smoke):
%   popSize, maxIter, archiveSize, k, localSearchTries
%   leaderBeta, rngSeed, correctEvery, finalCorrect

if nargin < 2 || isempty(opts), opts = struct(); end
opts = fill_opts(opts);

rng(opts.rngSeed);
N = ctx.N;
k = opts.k;
ctx.k = k;

% --- init ---
wolves = initializeWolves(N, k, ctx.S_B, ctx.potential, opts.popSize, opts.rngSeed);
for i = 1:opts.popSize
    wolves(i) = evaluateCandidate(wolves(i), ctx);
end
Archive = updateArchive([], wolves, opts.archiveSize);

history = struct('iter', {}, 'archiveSize', {}, 'bestFA', {}, 'bestCost', {}, ...
    'meanFA', {}, 'nEval', {});
nEval = opts.popSize;

fprintf('LSTM-MODGWO start: pop=%d iter=%d k=%d mode=%s archive=%d\n', ...
    opts.popSize, opts.maxIter, k, ctx.evalMode, opts.archiveSize);

t0 = tic;
for it = 1:opts.maxIter
    a = 2 - 2 * (it - 1) / max(opts.maxIter - 1, 1); % 2 -> 0
    leaders = selectLeaders(Archive, opts.leaderBeta);

    % optional periodic MC correction of archive under hybrid
    if strcmpi(ctx.evalMode, 'hybrid') && opts.correctEvery > 0 ...
            && mod(it, opts.correctEvery) == 0
        ctx.forceMC = true;
        for j = 1:numel(Archive)
            Archive(j) = evaluateCandidate(Archive(j), ctx);
            nEval = nEval + 1;
        end
        ctx.forceMC = false;
        Archive = updateArchive([], Archive, opts.archiveSize);
    end

    newWolves = wolves;
    for i = 1:opts.popSize
        Snew = discreteUpdate(wolves(i).Position, leaders, ctx.potential, ...
            ctx.S_B, N, a);
        cand = wolves(i);
        cand.Position = Snew;
        cand.SeedSet = Snew;
        cand = evaluateCandidate(cand, ctx);
        nEval = nEval + 1;

        if rand < opts.localSearchProb
            cand = localSearch(cand, ctx, opts.localSearchTries);
            nEval = nEval + opts.localSearchTries;
        end

        % greedy accept if dominates or random explore
        if dominates(cand.Cost, wolves(i).Cost) || rand < 0.1
            newWolves(i) = cand;
        else
            newWolves(i) = wolves(i);
        end
    end
    wolves = newWolves;
    Archive = updateArchive(Archive, wolves, opts.archiveSize);

    FA = [Archive.F_A];
    CC = [Archive.CostRaw];
    history(it).iter = it; %#ok<AGROW>
    history(it).archiveSize = numel(Archive);
    history(it).bestFA = max(FA);
    history(it).bestCost = min(CC);
    history(it).meanFA = mean(FA);
    history(it).nEval = nEval;

    if mod(it, max(1, floor(opts.maxIter / 5))) == 0 || it == opts.maxIter
        fprintf('  iter %3d/%d | archive=%d | maxFA=%.2f | minCost=%.3f | eval=%d | %.1fs\n', ...
            it, opts.maxIter, numel(Archive), max(FA), min(CC), nEval, toc(t0));
    end
end

% Final high-fidelity MC correction on archive
if opts.finalCorrect
    ctxMC = ctx;
    ctxMC.evalMode = 'mc';
    ctxMC.R_fast = opts.R_final;
    for j = 1:numel(Archive)
        Archive(j) = evaluateCandidate(Archive(j), ctxMC);
        nEval = nEval + 1;
    end
    Archive = updateArchive([], Archive, opts.archiveSize);
end

result = struct();
result.Archive = Archive;
result.wolves = wolves;
result.history = history;
result.opts = opts;
result.ctx = rmfield_safe(ctx, {'lstmModel', 'Network'}); % keep light log
result.nEval = nEval;
result.runtime = toc(t0);
result.S_B = ctx.S_B;

FA = [Archive.F_A];
CC = [Archive.CostRaw];
result.summary = struct('nArchive', numel(Archive), ...
    'maxFA', max(FA), 'minCost', min(CC), ...
    'meanFA', mean(FA), 'meanCost', mean(CC), 'nEval', nEval, ...
    'runtime', result.runtime);

fprintf('LSTM-MODGWO done: archive=%d maxFA=%.2f minCost=%.3f (%.1fs)\n', ...
    numel(Archive), max(FA), min(CC), result.runtime);
end

function opts = fill_opts(opts)
defs = struct( ...
    'popSize', 12, ...
    'maxIter', 8, ...
    'archiveSize', 20, ...
    'k', 5, ...
    'localSearchTries', 2, ...
    'localSearchProb', 0.3, ...
    'leaderBeta', 4, ...
    'rngSeed', 20261127, ...
    'correctEvery', 3, ...
    'finalCorrect', true, ...
    'R_final', 10);
fn = fieldnames(defs);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
        opts.(fn{i}) = defs.(fn{i});
    end
end
end

function tf = dominates(x, y)
x = x(:); y = y(:);
tf = all(x <= y) && any(x < y);
end

function s = rmfield_safe(s, fields)
for i = 1:numel(fields)
    if isfield(s, fields{i})
        s = rmfield(s, fields{i});
    end
end
end

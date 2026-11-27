function result = competitiveIC(Network, S_A, S_B, opts)
%COMPETITIVEIC  One-run competitive IC on directed weighted dynamic multilayer net.
%
% Inter-layer rule (confirmed):
%   Same physical node shares one global state across all layers.
%   Each layer provides parallel infection channels into inactive nodes.
%
% State coding: 0 inactive, 1 = A (ours), 2 = B (competitor)
%
% opts fields:
%   beta            scalar or 1xL vector (default 1)
%   layerNames      string array of layers to use
%   includeFacebook logical (default false)
%   maxRoundsPerSnapshot  (default 50)
%   epsVal          (default 1e-12)
%   recordEveryRound logical (default true)

if nargin < 4 || isempty(opts)
    opts = struct();
end
opts = fill_defaults(opts);

N = Network.numNodes;
T = Network.numSnapshots;
S_A = unique(S_A(:))';
S_B = unique(S_B(:))';
validate_seeds(S_A, S_B, N);

if isfield(opts, 'Pcache') && ~isempty(opts.Pcache)
    cache = opts.Pcache;
    layerIdx = cache.layerIdx;
    log1mP = cache.log1mP;
    betaUsed = cache.beta;
else
    cache = prepareICProbabilities(Network, opts);
    layerIdx = cache.layerIdx;
    log1mP = cache.log1mP;
    betaUsed = cache.beta;
end
Luse = numel(layerIdx);
beta = expand_beta(opts.beta, Network.numLayers);

state = zeros(N, 1);
state(S_A) = 1;
state(S_B) = 2;

sigmaA = [];
sigmaB = [];
snapOfStep = [];
roundOfStep = [];
step = 0;

for t = 1:T
    for r = 1:opts.maxRoundsPerSnapshot
        inactive = (state == 0);
        if ~any(inactive)
            break;
        end
        vA = double(state == 1);
        vB = double(state == 2);
        if ~any(vA) && ~any(vB)
            break;
        end

        sumLogFailA = zeros(N, 1);
        sumLogFailB = zeros(N, 1);
        for li = 1:Luse
            Lmat = log1mP{li, t};
            % Lmat(i,j)=log(1-p_ij); column j aggregates influencers i
            sumLogFailA = sumLogFailA + full(Lmat' * vA);
            sumLogFailB = sumLogFailB + full(Lmat' * vB);
        end

        probA = 1 - exp(sumLogFailA);
        probB = 1 - exp(sumLogFailB);
        probA(~inactive) = 0;
        probB(~inactive) = 0;
        probA = min(max(probA, 0), 1);
        probB = min(max(probB, 0), 1);

        [state, newlyA, newlyB] = updateNodeState(state, probA, probB, opts.epsVal);
        nNew = nnz(newlyA) + nnz(newlyB);

        if opts.recordEveryRound
            step = step + 1;
            sigmaA(step, 1) = nnz(state == 1); %#ok<AGROW>
            sigmaB(step, 1) = nnz(state == 2); %#ok<AGROW>
            snapOfStep(step, 1) = t; %#ok<AGROW>
            roundOfStep(step, 1) = r; %#ok<AGROW>
        end

        if nNew == 0
            break;
        end
    end

    if ~opts.recordEveryRound
        step = step + 1;
        sigmaA(step, 1) = nnz(state == 1);
        sigmaB(step, 1) = nnz(state == 2);
        snapOfStep(step, 1) = t;
        roundOfStep(step, 1) = 0;
    else
        % ensure at least one record per snapshot even if no inner rounds ran
        if isempty(snapOfStep) || snapOfStep(end) < t
            step = step + 1;
            sigmaA(step, 1) = nnz(state == 1);
            sigmaB(step, 1) = nnz(state == 2);
            snapOfStep(step, 1) = t;
            roundOfStep(step, 1) = 0;
        end
    end
end

result = struct();
result.state = state;
result.I_A = find(state == 1);
result.I_B = find(state == 2);
result.sigmaA = nnz(state == 1);
result.sigmaB = nnz(state == 2);
result.history.sigmaA = sigmaA;
result.history.sigmaB = sigmaB;
result.history.snapshot = snapOfStep;
result.history.round = roundOfStep;
result.meta.layerIdx = layerIdx;
result.meta.layerNames = string({Network.layer(layerIdx).name});
result.meta.beta = betaUsed;
result.meta.S_A = S_A;
result.meta.S_B = S_B;
result.meta.interLayer = "shared_physical_state";
end

function opts = fill_defaults(opts)
if ~isfield(opts, 'beta') || isempty(opts.beta), opts.beta = 1.0; end
if ~isfield(opts, 'includeFacebook'), opts.includeFacebook = false; end
if ~isfield(opts, 'maxRoundsPerSnapshot') || isempty(opts.maxRoundsPerSnapshot)
    opts.maxRoundsPerSnapshot = 50;
end
if ~isfield(opts, 'epsVal') || isempty(opts.epsVal), opts.epsVal = 1e-12; end
if ~isfield(opts, 'recordEveryRound'), opts.recordEveryRound = true; end
end

function validate_seeds(S_A, S_B, N)
if any(S_A < 1 | S_A > N) || any(S_B < 1 | S_B > N)
    error('competitiveIC:SeedRange', 'Seed indices must be in 1..N.');
end
if ~isempty(intersect(S_A, S_B))
    error('competitiveIC:SeedOverlap', 'S_A and S_B must be disjoint.');
end
if numel(unique(S_A)) ~= numel(S_A) || numel(unique(S_B)) ~= numel(S_B)
    error('competitiveIC:SeedDuplicate', 'Seed sets must not contain duplicates.');
end
end

function beta = expand_beta(betaIn, L)
if isscalar(betaIn)
    beta = repmat(double(betaIn), 1, L);
else
    beta = double(betaIn(:))';
    if numel(beta) ~= L
        error('competitiveIC:BetaSize', 'beta must be scalar or length numLayers.');
    end
end
end

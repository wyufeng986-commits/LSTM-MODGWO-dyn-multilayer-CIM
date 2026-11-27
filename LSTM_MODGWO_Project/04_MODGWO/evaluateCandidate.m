function wolf = evaluateCandidate(wolf, ctx)
%EVALUATECANDIDATE  Evaluate S_A under mc / lstm / hybrid modes.
%   ctx.evalMode: 'mc' | 'lstm' | 'hybrid'
%   Objectives: max F_A, min Cost_C4  -> Cost = [-F_A; Cost]

S_A = repairSeedSet(wolf.Position, ctx.k, ctx.N, ctx.S_B, ctx.potential);
wolf.Position = S_A;
wolf.SeedSet = S_A;
costVal = sum(ctx.costInfo.nodeCost(S_A));

mode = lower(string(ctx.evalMode));
switch mode
    case "mc"
        [FA, FB, detail] = eval_mc(S_A, ctx, ctx.R_fast);
        wolf.evalMode = 'mc';
    case "lstm"
        [FA, FB, detail] = eval_lstm(S_A, ctx);
        wolf.evalMode = 'lstm';
    case "hybrid"
        [FA, FB, detail] = eval_lstm(S_A, ctx);
        wolf.evalMode = 'lstm';
        if isfield(ctx, 'forceMC') && ctx.forceMC
            [FA, FB, detail] = eval_mc(S_A, ctx, ctx.R_correct);
            wolf.evalMode = 'mc';
        end
    otherwise
        error('evaluateCandidate:Mode', 'Unknown evalMode %s', ctx.evalMode);
end

wolf.F_A = FA;
wolf.F_B = FB;
wolf.CostRaw = costVal;
wolf.Cost = [-FA; costVal];
wolf.detail = detail;
end

function [FA, FB, detail] = eval_mc(S_A, ctx, R)
opts = ctx.icOpts;
opts.R = R;
opts.rngSeed = ctx.rngSeed + round(1e6 * rand());
obj = evaluateSeedObjectives(ctx.Network, S_A, ctx.S_B, ctx.costInfo, opts);
FA = obj.F_A;
FB = obj.F_B;
detail = obj;
end

function [FA, FB, detail] = eval_lstm(S_A, ctx)
% Short-horizon true IC then residual/absolute LSTM final prediction.
model = ctx.lstmModel;
r = model.historyLen;
icOpts = ctx.icOpts;
icOpts.recordEveryRound = false;
rng(ctx.rngSeed + sum(S_A) + 17);
out = competitiveIC(ctx.Network, S_A, ctx.S_B, icOpts);
sigA = out.history.sigmaA(:)';
sigB = out.history.sigmaB(:)';
T = ctx.Network.numSnapshots;
if numel(sigA) < T
    sigA = [sigA, repmat(sigA(end), 1, T - numel(sigA))];
    sigB = [sigB, repmat(sigB(end), 1, T - numel(sigB))];
else
    sigA = sigA(1:T); sigB = sigB(1:T);
end

zS = extractSeedFeatures(ctx.Network, S_A, ctx.S_B, ctx.costInfo, struct('includeFacebook', false));
Xdyn = extractTemporalFeatures(sigA, sigB, ctx.N);
Xin = [Xdyn(:, 1:r); repmat([zS; ctx.zG], 1, r)];

X = (Xin - model.muX) ./ model.sigX;
y = predict(model.net, {X});
if iscell(y), y = y{1}; end
y = y(:) .* model.sigY + model.muY;
y = y(1:min(2, numel(y)));
if numel(y) < 2, y(2) = 0; end

baseA = sigA(r); baseB = sigB(r);
if isfield(model, 'residual') && model.residual
    FA = baseA + y(1);
    FB = baseB + y(2);
else
    FA = y(1);
    FB = y(2);
end
FA = max(FA, numel(S_A));
FB = max(FB, numel(ctx.S_B));
N = ctx.N;
if FA + FB > N
    scale = N / (FA + FB);
    FA = FA * scale;
    FB = FB * scale;
end

detail = struct('baseA', baseA, 'baseB', baseB, 'y', y, ...
    'sigmaA_hist', sigA, 'sigmaB_hist', sigB);
end

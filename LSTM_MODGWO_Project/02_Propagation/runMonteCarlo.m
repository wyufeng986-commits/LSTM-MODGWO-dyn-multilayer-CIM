function mc = runMonteCarlo(Network, S_A, S_B, opts)
%RUNMONTECARLO  Repeated competitive IC with mean/std summaries.
%
% opts extra fields:
%   R            number of MC runs (default 20)
%   rngSeed      base seed (default 42); run r uses rngSeed+r-1
%   useParallel  logical (default false)

if nargin < 4 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'R') || isempty(opts.R), opts.R = 20; end
if ~isfield(opts, 'rngSeed') || isempty(opts.rngSeed), opts.rngSeed = 42; end
if ~isfield(opts, 'useParallel'), opts.useParallel = false; end

R = opts.R;
sigmaA = zeros(R, 1);
sigmaB = zeros(R, 1);
histA = cell(R, 1);
histB = cell(R, 1);

if opts.useParallel
    parfor r = 1:R
        localOpts = opts;
        rng(opts.rngSeed + r - 1);
        out = competitiveIC(Network, S_A, S_B, localOpts);
        sigmaA(r) = out.sigmaA;
        sigmaB(r) = out.sigmaB;
        histA{r} = out.history.sigmaA;
        histB{r} = out.history.sigmaB;
    end
else
    for r = 1:R
        rng(opts.rngSeed + r - 1);
        out = competitiveIC(Network, S_A, S_B, opts);
        sigmaA(r) = out.sigmaA;
        sigmaB(r) = out.sigmaB;
        histA{r} = out.history.sigmaA;
        histB{r} = out.history.sigmaB;
    end
end

[meanCurveA, meanCurveB, stdCurveA, stdCurveB] = average_curves(histA, histB);

mc = struct();
mc.R = R;
mc.F_A = mean(sigmaA);
mc.F_B = mean(sigmaB);
mc.std_A = std(sigmaA, 0);
mc.std_B = std(sigmaB, 0);
mc.sigmaA_runs = sigmaA;
mc.sigmaB_runs = sigmaB;
mc.meanCurveA = meanCurveA;
mc.meanCurveB = meanCurveB;
mc.stdCurveA = stdCurveA;
mc.stdCurveB = stdCurveB;
mc.opts = opts;
mc.S_A = S_A(:)';
mc.S_B = S_B(:)';
end

function [mA, mB, sA, sB] = average_curves(histA, histB)
R = numel(histA);
maxT = 0;
for r = 1:R
    maxT = max(maxT, numel(histA{r}));
end
if maxT == 0
    mA = []; mB = []; sA = []; sB = [];
    return;
end
A = nan(R, maxT);
B = nan(R, maxT);
for r = 1:R
    a = histA{r}(:)';
    b = histB{r}(:)';
    A(r, 1:numel(a)) = a;
    B(r, 1:numel(b)) = b;
    if numel(a) < maxT
        A(r, numel(a)+1:end) = a(end);
        B(r, numel(b)+1:end) = b(end);
    end
end
mA = mean(A, 1)';
mB = mean(B, 1)';
sA = std(A, 0, 1)';
sB = std(B, 0, 1)';
end

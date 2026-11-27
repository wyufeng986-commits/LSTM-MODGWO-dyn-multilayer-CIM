function result = experimentObjectiveConfirm(varargin)
%EXPERIMENTOBJECTIVECONFIRM  Stage-3 pre-experiment for bi-objective choice.
%   Samples many feasible S_A under fixed S_B and budget k, estimates (F_A,F_B),
%   analyzes correlation / Pareto trade-off, and recommends objective model.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('nSamples', 60, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('kA', 5, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('kB', 5, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('R', 15, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('beta', 1.0, @(x) isnumeric(x));
parser.addParameter('rngSeed', 20261127, @(x) isnumeric(x) && isscalar(x));
parser.parse(varargin{:});
opt = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opt.ProjectRoot)) == 0
    opt.ProjectRoot = fileparts(thisDir);
end
proj = char(opt.ProjectRoot);
addpath(fullfile(proj, '01_Data'));
addpath(fullfile(proj, '02_Propagation'));
addpath(fullfile(proj, '05_Experiments'));

Network = buildMultilayerNetwork('ProjectRoot', proj);
N = Network.numNodes;

% out-strength for fixing S_B
outStr = zeros(N, 1);
layerIdx = selectPropagationLayers(Network, struct('includeFacebook', false));
for li = 1:numel(layerIdx)
    l = layerIdx(li);
    for t = 1:Network.numSnapshots
        W = Network.layer(l).W{t};
        if ~issparse(W); W = sparse(W); end
        outStr = outStr + full(sum(W, 2));
    end
end

[SAlist, SB] = sampleRandomSeedSets(N, opt.kA, opt.kB, opt.nSamples, outStr, opt.rngSeed);

FA = zeros(opt.nSamples, 1);
FB = zeros(opt.nSamples, 1);
Cmin = zeros(opt.nSamples, 1); % optional layer coverage of A

mcOpts = struct('R', opt.R, 'beta', opt.beta, 'rngSeed', opt.rngSeed, ...
    'includeFacebook', false, 'maxRoundsPerSnapshot', 30, 'recordEveryRound', false);
mcOpts.Pcache = prepareICProbabilities(Network, mcOpts);

fprintf('Stage3 objective confirm: n=%d, kA=%d, kB=%d, R=%d, beta=%g\n', ...
    opt.nSamples, opt.kA, opt.kB, opt.R, opt.beta);
fprintf('Fixed S_B = %s\n', mat2str(SB));
fprintf('Layers: %s\n', strjoin(mcOpts.Pcache.layerNames, ', '));

tStart = tic;
for i = 1:opt.nSamples
    SA = SAlist{i};
    % use distinct rng stream offset per sample for MC stability comparison
    mcOpts.rngSeed = opt.rngSeed + 1000 * i;
    mc = runMonteCarlo(Network, SA, SB, mcOpts);
    FA(i) = mc.F_A;
    FB(i) = mc.F_B;

    % Cheap coverage proxy from last single run under same seed
    rng(mcOpts.rngSeed);
    one = competitiveIC(Network, SA, SB, mcOpts);
    Cmin(i) = layer_min_coverage(Network, one.state == 1, layerIdx);

    if mod(i, 5) == 0 || i == opt.nSamples
        fprintf('  [%3d/%3d] FA=%.2f FB=%.2f Cmin=%.3f  elapsed=%.1fs\n', ...
            i, opt.nSamples, FA(i), FB(i), Cmin(i), toc(tStart));
    end
end

pearson = corr(FA, FB, 'Type', 'Pearson');
spearman = corr(FA, FB, 'Type', 'Spearman');
[mask, frontIdx] = nonDominatedMaxMin(FA, FB);
nFront = numel(frontIdx);

% Synergy vs trade-off heuristic
% Positive corr(FA,FB): improving FA tends to also raise FB -> trade-off for (max FA, min FB)
% Strong negative corr: improving FA tends to suppress FB -> synergistic
if pearson >= 0.15 || (nFront >= max(3, round(0.05 * opt.nSamples)) && range(FA(frontIdx)) > 1e-6 && range(FB(frontIdx)) > 1e-6)
    recommendation = "primary_biobjective_FA_negFB";
    reason = "Observed dispersion on non-dominated set and/or non-negative corr(FA,FB), suggesting Pareto trade-off between max FA and min FB.";
elseif pearson <= -0.35 && nFront <= 3
    recommendation = "consider_FA_and_Cmin";
    reason = "Strong negative corr(FA,FB) with tiny front: objectives largely synergistic; consider (FA, Cmin) as alternative, keep FB as metric.";
else
    recommendation = "primary_biobjective_FA_negFB_with_caution";
    reason = "Mixed signal; keep primary (FA,-FB) for now and re-check with larger R / more samples.";
end

% Plots
figDir = fullfile(proj, '06_Results', 'figures');
tabDir = fullfile(proj, '06_Results', 'tables');
if ~exist(figDir, 'dir'); mkdir(figDir); end
if ~exist(tabDir, 'dir'); mkdir(tabDir); end

figure('Visible', 'off');
scatter(FA, FB, 28, [0.2 0.4 0.6], 'filled'); hold on;
scatter(FA(frontIdx), FB(frontIdx), 55, [0.85 0.2 0.15], 'filled');
xlabel('F_A (maximize)'); ylabel('F_B (minimize)');
title(sprintf('Objective scatter (n=%d,R=%d)  \\rho_{Pearson}=%.3f', opt.nSamples, opt.R, pearson));
legend({'samples', 'non-dominated'}, 'Location', 'best');
grid on;
fig1 = fullfile(figDir, 'stage3_objective_scatter.png');
saveas(gcf, fig1); close(gcf);

figure('Visible', 'off');
scatter(FA, Cmin, 28, [0.2 0.55 0.35], 'filled');
xlabel('F_A'); ylabel('C_{min} (A layer coverage)');
title('Optional alternative objective view');
grid on;
fig2 = fullfile(figDir, 'stage3_FA_Cmin_scatter.png');
saveas(gcf, fig2); close(gcf);

% Save table
tabPath = fullfile(tabDir, 'stage3_objective_samples.csv');
fid = fopen(tabPath, 'w');
fprintf(fid, 'idx,FA,FB,Cmin,is_nondominated,SA\n');
for i = 1:opt.nSamples
    fprintf(fid, '%d,%.6f,%.6f,%.6f,%d,"%s"\n', i, FA(i), FB(i), Cmin(i), mask(i), mat2str(SAlist{i}));
end
fclose(fid);

result = struct();
result.FA = FA;
result.FB = FB;
result.Cmin = Cmin;
result.frontIdx = frontIdx;
result.pearson = pearson;
result.spearman = spearman;
result.nFront = nFront;
result.recommendation = recommendation;
result.reason = reason;
result.SB = SB;
result.SAlist = SAlist;
result.opt = opt;
result.files.scatter = fig1;
result.files.fa_cmin = fig2;
result.files.table = tabPath;

outMat = fullfile(proj, '06_Results', 'logs', 'stage3_objective_confirm.mat');
if ~exist(fileparts(outMat), 'dir'); mkdir(fileparts(outMat)); end
save(outMat, 'result', 'Network', 'SB', 'FA', 'FB', 'Cmin', 'frontIdx', 'opt');
result.files.mat = outMat;

fprintf('\n=== Stage-3 summary ===\n');
fprintf('Pearson(FA,FB)=%.4f  Spearman=%.4f\n', pearson, spearman);
fprintf('Non-dominated count=%d / %d\n', nFront, opt.nSamples);
fprintf('Front FA range=[%.2f,%.2f], FB range=[%.2f,%.2f]\n', ...
    min(FA(frontIdx)), max(FA(frontIdx)), min(FB(frontIdx)), max(FB(frontIdx)));
fprintf('Recommendation: %s\n', recommendation);
fprintf('Reason: %s\n', reason);
fprintf('Saved %s\n', outMat);
end

function cmin = layer_min_coverage(Network, isA, layerIdx)
% Fraction of nodes that have incident edges in layer and are A-activated,
% then take min over layers. If a layer has no nodes with degree>0, skip.
N = Network.numNodes;
vals = [];
for li = 1:numel(layerIdx)
    l = layerIdx(li);
    deg = zeros(N, 1);
    for t = 1:Network.numSnapshots
        W = Network.layer(l).W{t};
        if ~issparse(W); W = sparse(W); end
        deg = deg + full(sum(W > 0, 2) + sum(W > 0, 1)');
    end
    activePool = find(deg > 0);
    if isempty(activePool)
        continue;
    end
    vals(end+1) = mean(isA(activePool)); %#ok<AGROW>
end
if isempty(vals)
    cmin = mean(isA);
else
    cmin = min(vals);
end
end

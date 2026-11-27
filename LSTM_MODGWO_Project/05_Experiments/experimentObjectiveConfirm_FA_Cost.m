function result = experimentObjectiveConfirm_FA_Cost(varargin)
%EXPERIMENTOBJECTIVECONFIRM_FA_COST  Confirm Pareto trade-off of (max FA, min Cost_C4).

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('nSamples', 60, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('kA', 5, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('kB', 5, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('R', 12, @(x) isnumeric(x) && isscalar(x));
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

costInfo = calculateNodeCostC4(Network, struct('includeFacebook', false));
outStr = costInfo.sOut; % for fixing competitor seeds

[SAlist, SB] = sampleRandomSeedSets(N, opt.kA, opt.kB, opt.nSamples, outStr, opt.rngSeed);

mcOpts = struct('R', opt.R, 'beta', opt.beta, 'rngSeed', opt.rngSeed, ...
    'includeFacebook', false, 'maxRoundsPerSnapshot', 30, 'recordEveryRound', false);
mcOpts.Pcache = prepareICProbabilities(Network, mcOpts);

FA = zeros(opt.nSamples, 1);
FB = zeros(opt.nSamples, 1);
Cost = zeros(opt.nSamples, 1);

fprintf('Stage3 FA-Cost confirm: n=%d kA=%d kB=%d R=%d beta=%g\n', ...
    opt.nSamples, opt.kA, opt.kB, opt.R, opt.beta);
fprintf('S_B=%s\n', mat2str(SB));
fprintf('Cost layers: %s\n', strjoin(costInfo.layerNames, ', '));

tStart = tic;
for i = 1:opt.nSamples
    mcOpts.rngSeed = opt.rngSeed + 1000 * i;
    obj = evaluateSeedObjectives(Network, SAlist{i}, SB, costInfo, mcOpts);
    FA(i) = obj.F_A;
    FB(i) = obj.F_B;
    Cost(i) = obj.Cost;
    if mod(i, 10) == 0 || i == opt.nSamples
        fprintf('  [%3d/%3d] FA=%.2f Cost=%.3f FB=%.2f  (%.1fs)\n', ...
            i, opt.nSamples, FA(i), Cost(i), FB(i), toc(tStart));
    end
end

% Non-dominated for max FA, min Cost
[mask, frontIdx] = nonDominatedMaxMin(FA, Cost);
nFront = numel(frontIdx);
pearson = corr(FA, Cost, 'Type', 'Pearson');
spearman = corr(FA, Cost, 'Type', 'Spearman');
pearsonFB = corr(FA, FB, 'Type', 'Pearson');

% Trade-off judgment
hasSpread = (range(FA(frontIdx)) > 1e-6) && (range(Cost(frontIdx)) > 1e-6);
if nFront >= max(3, round(0.05 * opt.nSamples)) && hasSpread
    recommendation = "accept_FA_CostC4";
    reason = "Non-dominated front has multiple points with spread in both FA and Cost; adopt max FA & min Cost_C4.";
elseif pearson > 0.2 && hasSpread
    recommendation = "accept_FA_CostC4";
    reason = "Positive corr(FA,Cost) indicates higher influence tends to cost more; bi-objective is meaningful.";
else
    recommendation = "review_FA_CostC4";
    reason = "Front/correlation signal is weak; inspect scatter before locking.";
end

figDir = fullfile(proj, '06_Results', 'figures');
tabDir = fullfile(proj, '06_Results', 'tables');
logDir = fullfile(proj, '06_Results', 'logs');
if ~exist(figDir, 'dir'); mkdir(figDir); end
if ~exist(tabDir, 'dir'); mkdir(tabDir); end
if ~exist(logDir, 'dir'); mkdir(logDir); end

figure('Visible', 'off');
scatter(Cost, FA, 28, [0.25 0.45 0.65], 'filled'); hold on;
scatter(Cost(frontIdx), FA(frontIdx), 60, [0.85 0.2 0.15], 'filled');
xlabel('Cost_{C4} (minimize)');
ylabel('F_A (maximize)');
title(sprintf('(F_A, Cost) n=%d  \\rho=%.3f  front=%d', opt.nSamples, pearson, nFront));
legend({'samples','non-dominated'}, 'Location', 'best');
grid on;
fig1 = fullfile(figDir, 'stage3_FA_Cost_scatter.png');
saveas(gcf, fig1); close(gcf);

figure('Visible', 'off');
scatter(FA, FB, 28, Cost, 'filled'); colorbar;
xlabel('F_A'); ylabel('F_B (metric)');
title('FA vs FB colored by Cost_{C4}');
grid on;
fig2 = fullfile(figDir, 'stage3_FA_FB_colored_by_Cost.png');
saveas(gcf, fig2); close(gcf);

tabPath = fullfile(tabDir, 'stage3_FA_Cost_samples.csv');
fid = fopen(tabPath, 'w');
fprintf(fid, 'idx,FA,Cost,FB,is_nondominated,SA\n');
for i = 1:opt.nSamples
    fprintf(fid, '%d,%.6f,%.6f,%.6f,%d,"%s"\n', i, FA(i), Cost(i), FB(i), mask(i), mat2str(SAlist{i}));
end
fclose(fid);

result = struct();
result.FA = FA;
result.Cost = Cost;
result.FB = FB;
result.frontIdx = frontIdx;
result.nFront = nFront;
result.pearson_FA_Cost = pearson;
result.spearman_FA_Cost = spearman;
result.pearson_FA_FB = pearsonFB;
result.recommendation = recommendation;
result.reason = reason;
result.SB = SB;
result.SAlist = SAlist;
result.costInfo = costInfo;
result.opt = opt;
result.files = struct('scatter', fig1, 'fa_fb', fig2, 'table', tabPath);

outMat = fullfile(logDir, 'stage3_FA_Cost_confirm.mat');
save(outMat, 'result', 'Network', 'SB', 'FA', 'Cost', 'FB', 'frontIdx', 'costInfo', 'opt');
result.files.mat = outMat;

fprintf('\n=== FA-Cost summary ===\n');
fprintf('Pearson(FA,Cost)=%.4f  Spearman=%.4f\n', pearson, spearman);
fprintf('Pearson(FA,FB)=%.4f (metric only)\n', pearsonFB);
fprintf('Non-dominated=%d/%d\n', nFront, opt.nSamples);
fprintf('Front FA=[%.2f,%.2f], Cost=[%.3f,%.3f]\n', ...
    min(FA(frontIdx)), max(FA(frontIdx)), min(Cost(frontIdx)), max(Cost(frontIdx)));
fprintf('Recommendation: %s\n%s\n', recommendation, reason);
fprintf('Saved %s\n', outMat);
end

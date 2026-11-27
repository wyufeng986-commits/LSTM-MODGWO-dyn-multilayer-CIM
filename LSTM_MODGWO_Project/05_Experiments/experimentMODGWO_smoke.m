function result = experimentMODGWO_smoke(varargin)
%EXPERIMENTMODGWO_SMOKE  Stage-5 smoke: discrete MODGWO + optional LSTM.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('evalMode', 'hybrid', @(x) ischar(x) || isstring(x));
parser.addParameter('popSize', 10, @isscalar);
parser.addParameter('maxIter', 6, @isscalar);
parser.addParameter('archiveSize', 15, @isscalar);
parser.addParameter('kA', 5, @isscalar);
parser.addParameter('kB', 5, @isscalar);
parser.addParameter('R_fast', 3, @isscalar);
parser.addParameter('R_final', 8, @isscalar);
parser.addParameter('beta', 1.0, @(x) isnumeric(x));
parser.addParameter('rngSeed', 20261127, @isscalar);
parser.addParameter('finalCorrect', true, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
opt = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opt.ProjectRoot)) == 0
    opt.ProjectRoot = fileparts(thisDir);
end
proj = char(opt.ProjectRoot);
addpath(genpath(proj));

Network = buildMultilayerNetwork('ProjectRoot', proj);
N = Network.numNodes;
costInfo = calculateNodeCostC4(Network, struct('includeFacebook', false));
potential = calculateNodePotential(costInfo, []);
zG = extractNetworkFeatures(Network, struct('includeFacebook', false));

[~, SB] = sampleRandomSeedSets(N, opt.kA, opt.kB, 1, costInfo.sOut, opt.rngSeed);
potential = calculateNodePotential(costInfo, SB);

icOpts = struct('beta', opt.beta, 'includeFacebook', false, ...
    'maxRoundsPerSnapshot', 30, 'recordEveryRound', false, ...
    'R', opt.R_fast, 'rngSeed', opt.rngSeed);
icOpts.Pcache = prepareICProbabilities(Network, icOpts);

ctx = struct();
ctx.Network = Network;
ctx.N = N;
ctx.k = opt.kA;
ctx.S_B = SB;
ctx.costInfo = costInfo;
ctx.potential = potential;
ctx.zG = zG;
ctx.icOpts = icOpts;
ctx.evalMode = char(opt.evalMode);
ctx.R_fast = opt.R_fast;
ctx.R_correct = opt.R_final;
ctx.rngSeed = opt.rngSeed;
ctx.forceMC = false;

% Load residual LSTM if needed
if any(strcmpi(ctx.evalMode, {'lstm', 'hybrid'}))
    modelFile = fullfile(proj, '03_LSTM', 'models', 'model_lstm_resid_h4.mat');
    if ~isfile(modelFile)
        warning('experimentMODGWO_smoke:NoLSTM', ...
            'Residual LSTM not found, fallback to mc.');
        ctx.evalMode = 'mc';
    else
        S = load(modelFile, 'model');
        model = S.model;
        model.residual = true;
        ctx.lstmModel = model;
    end
end

algoOpts = struct( ...
    'popSize', opt.popSize, ...
    'maxIter', opt.maxIter, ...
    'archiveSize', opt.archiveSize, ...
    'k', opt.kA, ...
    'localSearchTries', 2, ...
    'localSearchProb', 0.35, ...
    'rngSeed', opt.rngSeed, ...
    'correctEvery', 2, ...
    'finalCorrect', logical(opt.finalCorrect), ...
    'R_final', opt.R_final);

fprintf('=== Stage5 MODGWO smoke ===\n');
fprintf('N=%d kA=%d kB=%d mode=%s pop=%d iter=%d\n', ...
    N, opt.kA, opt.kB, ctx.evalMode, opt.popSize, opt.maxIter);
fprintf('S_B=%s\n', mat2str(SB));

out = LSTM_MODGWO(ctx, algoOpts);

% Sanity checks
Archive = out.Archive;
assert(~isempty(Archive), 'Archive empty');
for i = 1:numel(Archive)
    S = Archive(i).Position;
    assert(numel(unique(S)) == opt.kA, 'Invalid |S_A|');
    assert(isempty(intersect(S, SB)), 'S_A overlaps S_B');
end
FA = [Archive.F_A];
CC = [Archive.CostRaw];
[mask, frontIdx] = nonDominatedMaxMin(FA, CC);
assert(numel(frontIdx) == numel(Archive), 'Archive contains dominated solutions');

% Save
tabDir = fullfile(proj, '06_Results', 'tables');
figDir = fullfile(proj, '06_Results', 'figures');
logDir = fullfile(proj, '06_Results', 'logs');
if ~exist(tabDir, 'dir'), mkdir(tabDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(logDir, 'dir'), mkdir(logDir); end

csvFile = fullfile(tabDir, 'stage5_modgwo_smoke.csv');
fid = fopen(csvFile, 'w');
fprintf(fid, 'idx,FA,Cost,FB,evalMode,seeds\n');
for i = 1:numel(Archive)
    fprintf(fid, '%d,%.6f,%.6f,%.6f,%s,\"%s\"\n', i, Archive(i).F_A, ...
        Archive(i).CostRaw, Archive(i).F_B, Archive(i).evalMode, mat2str(Archive(i).Position));
end
fclose(fid);

figure('Visible', 'off');
scatter(CC, FA, 48, 'filled'); grid on;
xlabel('Cost_{C4}'); ylabel('F_A');
title(sprintf('Stage5 smoke Pareto (n=%d, mode=%s)', numel(Archive), ctx.evalMode));
figFile = fullfile(figDir, 'stage5_modgwo_smoke_pareto.png');
saveas(gcf, figFile); close(gcf);

result = out;
result.files = struct('csv', csvFile, 'fig', figFile);
result.opt = opt;
result.frontMask = mask;
matFile = fullfile(logDir, 'stage5_modgwo_smoke.mat');
save(matFile, 'result', 'Archive', 'opt', '-v7.3');
result.files.mat = matFile;

fprintf('\nSmoke OK: archive=%d maxFA=%.2f minCost=%.3f\n', ...
    numel(Archive), max(FA), min(CC));
fprintf('Saved %s\n', matFile);
end

function result = experimentMODGWO_formal(varargin)
%EXPERIMENTMODGWO_FORMAL  Formal Stage-5 MODGWO run with logging & artifacts.
%
% Intended to be executed on a dedicated machine, e.g.:
%   result = experimentMODGWO_formal('preset','formal');
%   result = experimentMODGWO_formal('preset','medium');  % remote dry-run first
%
% Name-value overrides are allowed after preset load.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('preset', 'formal', @(x) ischar(x) || isstring(x));
parser.addParameter('evalMode', '', @(x) ischar(x) || isstring(x));
parser.addParameter('popSize', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('maxIter', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('archiveSize', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('kA', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('kB', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('R_fast', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('R_final', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('correctEvery', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('beta', [], @(x) isempty(x) || isnumeric(x));
parser.addParameter('rngSeed', [], @(x) isempty(x) || isscalar(x));
parser.addParameter('finalCorrect', [], @(x) isempty(x) || islogical(x) || isnumeric(x));
parser.addParameter('SkipPreflight', false, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
user = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(user.ProjectRoot)) == 0
    user.ProjectRoot = fileparts(thisDir);
end
proj = char(user.ProjectRoot);
addpath(genpath(proj));

cfg = config_modgwo_formal(user.preset);
cfg = overlay_cfg(cfg, user);
cfg.ProjectRoot = proj;
cfg.tag = sprintf('%s_p%d_i%d_k%d_%s', cfg.preset, cfg.popSize, ...
    cfg.maxIter, cfg.kA, cfg.evalMode);

% Output dirs
tabDir = fullfile(proj, '06_Results', 'tables');
figDir = fullfile(proj, '06_Results', 'figures');
logDir = fullfile(proj, '06_Results', 'logs');
ensure_dir(tabDir); ensure_dir(figDir); ensure_dir(logDir);

stamp = datestr(now, 'yyyymmdd_HHMMSS');
runId = sprintf('stage5_modgwo_%s_%s', cfg.tag, stamp);
diaryFile = fullfile(logDir, [runId, '.diary.txt']);
cfgFile = fullfile(logDir, [runId, '_config.json']);
matFile = fullfile(logDir, [runId, '.mat']);
csvFile = fullfile(tabDir, [runId, '_archive.csv']);
histCsv = fullfile(tabDir, [runId, '_history.csv']);
figPareto = fullfile(figDir, [runId, '_pareto.png']);
figHist = fullfile(figDir, [runId, '_history.png']);

diary(diaryFile);
cleanupObj = onCleanup(@() safe_diary_off()); %#ok<NASGU>

fprintf('============================================================\n');
fprintf(' Stage-5 formal MODGWO\n');
fprintf(' runId : %s\n', runId);
fprintf(' root  : %s\n', proj);
fprintf('============================================================\n');
disp(cfg);

if ~user.SkipPreflight
    preflight_modgwo(proj, cfg);
end
write_cfg_json(cfgFile, cfg, runId);

% --- build context ---
Network = buildMultilayerNetwork('ProjectRoot', proj);
N = Network.numNodes;
costInfo = calculateNodeCostC4(Network, struct('includeFacebook', false));
zG = extractNetworkFeatures(Network, struct('includeFacebook', false));
[~, SB] = sampleRandomSeedSets(N, cfg.kA, cfg.kB, 1, costInfo.sOut, cfg.rngSeed);
potential = calculateNodePotential(costInfo, SB);

icOpts = struct('beta', cfg.beta, 'includeFacebook', false, ...
    'maxRoundsPerSnapshot', 30, 'recordEveryRound', false, ...
    'R', cfg.R_fast, 'rngSeed', cfg.rngSeed);
icOpts.Pcache = prepareICProbabilities(Network, icOpts);

ctx = struct();
ctx.Network = Network;
ctx.N = N;
ctx.k = cfg.kA;
ctx.S_B = SB;
ctx.costInfo = costInfo;
ctx.potential = potential;
ctx.zG = zG;
ctx.icOpts = icOpts;
ctx.evalMode = char(cfg.evalMode);
ctx.R_fast = cfg.R_fast;
ctx.R_correct = cfg.R_final;
ctx.rngSeed = cfg.rngSeed;
ctx.forceMC = false;

if any(strcmpi(ctx.evalMode, {'lstm', 'hybrid'}))
    modelFile = fullfile(proj, cfg.lstmModelRel);
    S = load(modelFile, 'model');
    model = S.model;
    model.residual = true;
    ctx.lstmModel = model;
    fprintf('Loaded LSTM: %s (residual=%d, r=%d, feat=%d)\n', ...
        modelFile, model.residual, model.historyLen, model.featureDim);
end

algoOpts = struct( ...
    'popSize', cfg.popSize, ...
    'maxIter', cfg.maxIter, ...
    'archiveSize', cfg.archiveSize, ...
    'k', cfg.kA, ...
    'localSearchTries', cfg.localSearchTries, ...
    'localSearchProb', cfg.localSearchProb, ...
    'leaderBeta', cfg.leaderBeta, ...
    'rngSeed', cfg.rngSeed, ...
    'correctEvery', cfg.correctEvery, ...
    'finalCorrect', logical(cfg.finalCorrect), ...
    'R_final', cfg.R_final);

fprintf('S_B = %s\n', mat2str(SB));
fprintf('Start optimization at %s\n', datestr(now));
tWall = tic;
out = LSTM_MODGWO(ctx, algoOpts);
wallTime = toc(tWall);
fprintf('Optimization wall time: %.1f s (%.2f min)\n', wallTime, wallTime / 60);

Archive = out.Archive;
assert(~isempty(Archive), 'Archive empty after formal run.');
for i = 1:numel(Archive)
    Sset = Archive(i).Position;
    assert(numel(unique(Sset)) == cfg.kA, 'Invalid |S_A| at archive %d', i);
    assert(isempty(intersect(Sset, SB)), 'S_A overlaps S_B at archive %d', i);
end
FA = [Archive.F_A];
CC = [Archive.CostRaw];
FB = [Archive.F_B];
[mask, frontIdx] = nonDominatedMaxMin(FA, CC);
assert(numel(frontIdx) == numel(Archive), 'Dominated solutions remain in archive.');

% --- save archive csv ---
fid = fopen(csvFile, 'w');
fprintf(fid, 'idx,FA,Cost,FB,evalMode,seeds\n');
for i = 1:numel(Archive)
    fprintf(fid, '%d,%.8f,%.8f,%.8f,%s,"%s"\n', i, Archive(i).F_A, ...
        Archive(i).CostRaw, Archive(i).F_B, Archive(i).evalMode, ...
        mat2str(Archive(i).Position));
end
fclose(fid);

% --- save history csv ---
hist = out.history;
fid = fopen(histCsv, 'w');
fprintf(fid, 'iter,archiveSize,bestFA,bestCost,meanFA,nEval\n');
for i = 1:numel(hist)
    fprintf(fid, '%d,%d,%.8f,%.8f,%.8f,%d\n', hist(i).iter, hist(i).archiveSize, ...
        hist(i).bestFA, hist(i).bestCost, hist(i).meanFA, hist(i).nEval);
end
fclose(fid);

% --- figures ---
figure('Visible', 'off');
scatter(CC, FA, 56, 'filled'); grid on;
xlabel('Cost_{C4}'); ylabel('F_A');
title(sprintf('Formal Pareto | %s | n=%d', cfg.tag, numel(Archive)));
saveas(gcf, figPareto); close(gcf);

figure('Visible', 'off');
subplot(2,1,1);
plot([hist.iter], [hist.bestFA], '-o', 'LineWidth', 1.2); grid on;
ylabel('best F_A (approx/corrected)'); xlabel('iter');
subplot(2,1,2);
plot([hist.iter], [hist.archiveSize], '-o', 'LineWidth', 1.2); grid on;
ylabel('archive size'); xlabel('iter');
sgtitle(cfg.tag, 'Interpreter', 'none');
saveas(gcf, figHist); close(gcf);

% --- package result ---
result = out;
result.runId = runId;
result.cfg = cfg;
result.S_B = SB;
result.frontMask = mask;
result.wallTime = wallTime;
result.files = struct( ...
    'diary', diaryFile, ...
    'config', cfgFile, ...
    'mat', matFile, ...
    'archiveCsv', csvFile, ...
    'historyCsv', histCsv, ...
    'figPareto', figPareto, ...
    'figHistory', figHist);

% Keep Network out of mat to reduce size; store summary only
netInfo = struct('numNodes', N, 'numLayers', Network.numLayers, ...
    'numSnapshots', Network.numSnapshots, 'layerNames', {Network.layerNames});
save(matFile, 'result', 'Archive', 'cfg', 'SB', 'netInfo', '-v7.3');

fprintf('\n=== Formal run finished ===\n');
fprintf('archive=%d  maxFA=%.4f  minCost=%.4f  meanFA=%.4f\n', ...
    numel(Archive), max(FA), min(CC), mean(FA));
fprintf('nEval=%d  wall=%.1fs\n', result.nEval, wallTime);
fprintf('MAT : %s\n', matFile);
fprintf('CSV : %s\n', csvFile);
fprintf('LOG : %s\n', diaryFile);
end

function cfg = overlay_cfg(cfg, user)
fields = {'evalMode','popSize','maxIter','archiveSize','kA','kB', ...
    'R_fast','R_final','correctEvery','beta','rngSeed','finalCorrect', ...
    'localSearchTries','localSearchProb','leaderBeta'};
for i = 1:numel(fields)
    f = fields{i};
    if isfield(user, f) && ~isempty(user.(f))
        val = user.(f);
        if isstring(val), val = char(val); end
        cfg.(f) = val;
    end
end
end

function preflight_modgwo(proj, cfg)
need = { ...
    fullfile(proj, '01_Data', 'processed', 'CNS_daily_multilayer.mat'); ...
    fullfile(proj, '04_MODGWO', 'LSTM_MODGWO.m'); ...
    };
if any(strcmpi(cfg.evalMode, {'lstm','hybrid'}))
    need{end+1} = fullfile(proj, cfg.lstmModelRel); %#ok<AGROW>
end
missing = {};
for i = 1:numel(need)
    if ~isfile(need{i})
        missing{end+1} = need{i}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    fprintf(2, 'Preflight FAILED. Missing files:\n');
    for i = 1:numel(missing)
        fprintf(2, '  - %s\n', missing{i});
    end
    error('experimentMODGWO_formal:Preflight', ...
        'Copy the full LSTM_MODGWO_Project folder including processed data and LSTM model.');
end

v = ver;
hasDL = any(strcmp({v.Name}, 'Deep Learning Toolbox'));
hasStat = any(strcmp({v.Name}, 'Statistics and Machine Learning Toolbox'));
fprintf('MATLAB %s | DeepLearning=%d | Stats=%d\n', version, hasDL, hasStat);
if any(strcmpi(cfg.evalMode, {'lstm','hybrid'})) && ~hasDL
    error('experimentMODGWO_formal:Toolbox', ...
        'Deep Learning Toolbox required for hybrid/lstm mode.');
end
fprintf('Preflight OK.\n');
end

function write_cfg_json(path, cfg, runId)
% Minimal JSON without toolbox dependency.
fid = fopen(path, 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "runId": "%s",\n', escape_json(runId));
fprintf(fid, '  "preset": "%s",\n', escape_json(cfg.preset));
fprintf(fid, '  "tag": "%s",\n', escape_json(cfg.tag));
fprintf(fid, '  "evalMode": "%s",\n', escape_json(cfg.evalMode));
fprintf(fid, '  "popSize": %d,\n', cfg.popSize);
fprintf(fid, '  "maxIter": %d,\n', cfg.maxIter);
fprintf(fid, '  "archiveSize": %d,\n', cfg.archiveSize);
fprintf(fid, '  "kA": %d,\n', cfg.kA);
fprintf(fid, '  "kB": %d,\n', cfg.kB);
fprintf(fid, '  "R_fast": %d,\n', cfg.R_fast);
fprintf(fid, '  "R_final": %d,\n', cfg.R_final);
fprintf(fid, '  "correctEvery": %d,\n', cfg.correctEvery);
fprintf(fid, '  "beta": %.6g,\n', cfg.beta);
fprintf(fid, '  "rngSeed": %d,\n', cfg.rngSeed);
fprintf(fid, '  "finalCorrect": %s,\n', lower(string(logical(cfg.finalCorrect))));
fprintf(fid, '  "note": "%s"\n', escape_json(cfg.note));
fprintf(fid, '}\n');
fclose(fid);
end

function s = escape_json(s)
s = char(string(s));
s = strrep(s, '\', '\\');
s = strrep(s, '"', '\"');
s = strrep(s, newline, ' ');
end

function ensure_dir(p)
if ~exist(p, 'dir'), mkdir(p); end
end

function safe_diary_off()
try
    diary off;
catch
end
end

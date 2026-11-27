function result = experimentLSTM(varargin)
%EXPERIMENTLSTM  Stage-4 sample generation, LSTM/GRU train, baseline comparison.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('nSeedCombos', 36, @isscalar);
parser.addParameter('nTrajPerCombo', 2, @isscalar);
parser.addParameter('historyLen', 8, @isscalar);
parser.addParameter('maxEpochs', 30, @isscalar);
parser.addParameter('hiddenUnits', 48, @isscalar);
parser.parse(varargin{:});
opt = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opt.ProjectRoot)) == 0
    opt.ProjectRoot = fileparts(thisDir);
end
proj = char(opt.ProjectRoot);
addpath(fullfile(proj, '01_Data'));
addpath(fullfile(proj, '02_Propagation'));
addpath(fullfile(proj, '03_LSTM'));
addpath(fullfile(proj, '05_Experiments'));

fprintf('=== Stage4 LSTM experiment ===\n');
sampleFile = fullfile(proj, '03_LSTM', 'models', 'lstm_raw_samples.mat');
if isfile(sampleFile)
    fprintf('Reusing existing samples: %s\n', sampleFile);
    S = load(sampleFile, 'samples');
    samples = S.samples;
    samples.file = sampleFile;
else
    samples = generateLSTMSamples('ProjectRoot', proj, ...
        'nSeedCombos', opt.nSeedCombos, ...
        'nTrajPerCombo', opt.nTrajPerCombo, ...
        'historyLen', opt.historyLen);
end

data = prepareSequenceData(samples, 'historyLen', opt.historyLen);
fprintf('Split train/val/test combos: %d/%d/%d | hist=%d horizon=%d\n', ...
    numel(data.trainCombos), numel(data.valCombos), numel(data.testCombos), ...
    data.historyLen, data.horizon);

lstmFile = fullfile(proj, '03_LSTM', 'models', 'model_lstm.mat');
gruFile = fullfile(proj, '03_LSTM', 'models', 'model_gru.mat');
if isfile(lstmFile)
    fprintf('Reusing LSTM model: %s\n', lstmFile);
    tmp = load(lstmFile, 'model'); modelLSTM = tmp.model; modelLSTM.file = lstmFile;
else
    modelLSTM = trainLSTM(data, 'modelType', 'lstm', 'maxEpochs', opt.maxEpochs, ...
        'hiddenUnits', opt.hiddenUnits, 'ProjectRoot', proj);
end
if isfile(gruFile)
    fprintf('Reusing GRU model: %s\n', gruFile);
    tmp = load(gruFile, 'model'); modelGRU = tmp.model; modelGRU.file = gruFile;
else
    modelGRU = trainLSTM(data, 'modelType', 'gru', 'maxEpochs', opt.maxEpochs, ...
        'hiddenUnits', opt.hiddenUnits, 'ProjectRoot', proj);
end

mLSTM = evaluateLSTM(modelLSTM, data, 'test');
mGRU = evaluateLSTM(modelGRU, data, 'test');
mPers = baselinePredictors(data, 'persistence');
mLin = baselinePredictors(data, 'linear');
try
    mSVR = baselinePredictors(data, 'svr');
catch ME
    warning('SVR baseline failed: %s', ME.message);
    mSVR = struct('name', 'svr', 'MAE', nan, 'RMSE', nan, 'R2', nan, ...
        'finalA_MAE', nan, 'finalB_MAE', nan, 'n', 0);
end

% horizon sensitivity with LSTM: reuse same model metrics by truncating? 
% quick retrain is expensive; report final-step MAE vs step index
H = modelLSTM.horizon;
stepMAE_A = zeros(H, 1); stepMAE_B = zeros(H, 1);
for t = 1:H
    stepMAE_A(t) = mean(abs(mLSTM.Yp(1, t, :) - mLSTM.Yt(1, t, :)), 'all');
    stepMAE_B(t) = mean(abs(mLSTM.Yp(2, t, :) - mLSTM.Yt(2, t, :)), 'all');
end

figDir = fullfile(proj, '06_Results', 'figures');
tabDir = fullfile(proj, '06_Results', 'tables');
logDir = fullfile(proj, '06_Results', 'logs');
if ~exist(figDir, 'dir'); mkdir(figDir); end
if ~exist(tabDir, 'dir'); mkdir(tabDir); end
if ~exist(logDir, 'dir'); mkdir(logDir); end

% comparison table
tab = fullfile(tabDir, 'stage4_lstm_comparison.csv');
fid = fopen(tab, 'w');
fprintf(fid, 'model,MAE,RMSE,R2,finalA_MAE,finalB_MAE,n\n');
rows = {mPers, mLin, mSVR, mGRU, mLSTM};
names = {'persistence', 'linear', 'svr', 'gru', 'lstm'};
for i = 1:numel(rows)
    m = rows{i};
    fprintf(fid, '%s,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n', names{i}, ...
        m.MAE, m.RMSE, m.R2, m.finalA_MAE, m.finalB_MAE, m.n);
end
fclose(fid);

figure('Visible', 'off');
plot(1:H, stepMAE_A, '-o', 'DisplayName', '\sigma_A'); hold on;
plot(1:H, stepMAE_B, '-s', 'DisplayName', '\sigma_B');
xlabel('Prediction step (after history)'); ylabel('MAE');
title('LSTM error vs prediction horizon');
legend('Location', 'best'); grid on;
fig1 = fullfile(figDir, 'stage4_lstm_horizon_mae.png');
saveas(gcf, fig1); close(gcf);

% example prediction curve (future horizon only)
figure('Visible', 'off');
ex = 1;
tt = data.historyLen + (1:H);
ytA = squeeze(mLSTM.Yt(1, :, ex)); ytA = ytA(:)';
ypA = squeeze(mLSTM.Yp(1, :, ex)); ypA = ypA(:)';
ytB = squeeze(mLSTM.Yt(2, :, ex)); ytB = ytB(:)';
ypB = squeeze(mLSTM.Yp(2, :, ex)); ypB = ypB(:)';
plot(tt, ytA, 'b-', 'LineWidth', 1.5); hold on;
plot(tt, ypA, 'b--', 'LineWidth', 1.5);
plot(tt, ytB, 'r-', 'LineWidth', 1.5);
plot(tt, ypB, 'r--', 'LineWidth', 1.5);
xlabel('t'); ylabel('spread');
legend('\sigma_A true','\sigma_A pred','\sigma_B true','\sigma_B pred', 'Location', 'best');
title('Example LSTM multi-step prediction'); grid on;
fig2 = fullfile(figDir, 'stage4_lstm_example_curve.png');
saveas(gcf, fig2); close(gcf);

result = struct();
result.samplesFile = samples.file;
result.metrics = struct('persistence', mPers, 'linear', mLin, 'svr', mSVR, ...
    'gru', mGRU, 'lstm', mLSTM);
result.stepMAE_A = stepMAE_A;
result.stepMAE_B = stepMAE_B;
result.files = struct('table', tab, 'horizonFig', fig1, 'exampleFig', fig2);
result.modelLSTM = modelLSTM.file;
result.modelGRU = modelGRU.file;

outMat = fullfile(logDir, 'stage4_lstm_experiment.mat');
save(outMat, 'result', 'data', 'mLSTM', 'mGRU', 'mPers', 'mLin', 'mSVR', 'opt');
result.files.mat = outMat;

fprintf('\n=== Stage4 summary (test) ===\n');
fprintf('%-12s MAE=%.3f RMSE=%.3f R2=%.3f finalA_MAE=%.2f\n', 'persistence', mPers.MAE, mPers.RMSE, mPers.R2, mPers.finalA_MAE);
fprintf('%-12s MAE=%.3f RMSE=%.3f R2=%.3f finalA_MAE=%.2f\n', 'linear', mLin.MAE, mLin.RMSE, mLin.R2, mLin.finalA_MAE);
fprintf('%-12s MAE=%.3f RMSE=%.3f R2=%.3f finalA_MAE=%.2f\n', 'svr', mSVR.MAE, mSVR.RMSE, mSVR.R2, mSVR.finalA_MAE);
fprintf('%-12s MAE=%.3f RMSE=%.3f R2=%.3f finalA_MAE=%.2f\n', 'gru', mGRU.MAE, mGRU.RMSE, mGRU.R2, mGRU.finalA_MAE);
fprintf('%-12s MAE=%.3f RMSE=%.3f R2=%.3f finalA_MAE=%.2f\n', 'lstm', mLSTM.MAE, mLSTM.RMSE, mLSTM.R2, mLSTM.finalA_MAE);
fprintf('Saved %s\n', outMat);
end

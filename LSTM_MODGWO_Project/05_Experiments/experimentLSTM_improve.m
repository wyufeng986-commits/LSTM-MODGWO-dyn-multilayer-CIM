function result = experimentLSTM_improve(varargin)
%EXPERIMENTLSTM_IMPROVE  Improve FA prediction: more data + residual final-target.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('nSeedCombos', 80, @isscalar);
parser.addParameter('nTrajPerCombo', 3, @isscalar);
parser.addParameter('historyLen', 4, @isscalar);
parser.addParameter('maxEpochs', 40, @isscalar);
parser.addParameter('hiddenUnits', 64, @isscalar);
parser.addParameter('forceRegen', true, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
opt = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opt.ProjectRoot)) == 0
    opt.ProjectRoot = fileparts(thisDir);
end
proj = char(opt.ProjectRoot);
addpath(genpath(proj));

sampleFile = fullfile(proj, '03_LSTM', 'models', 'lstm_raw_samples_v2.mat');
if opt.forceRegen || ~isfile(sampleFile)
    samples = generateLSTMSamples('ProjectRoot', proj, ...
        'nSeedCombos', opt.nSeedCombos, ...
        'nTrajPerCombo', opt.nTrajPerCombo, ...
        'historyLen', opt.historyLen, ...
        'rngSeed', 20261128);
    % relocate default save
    if isfield(samples, 'file') && isfile(samples.file)
        copyfile(samples.file, sampleFile);
        samples.file = sampleFile;
    end
else
    S = load(sampleFile, 'samples'); samples = S.samples; samples.file = sampleFile;
end

data = prepareSequenceData_final(samples, 'historyLen', opt.historyLen, 'residual', true);
fprintf('Improved setup: combos train/val/test=%d/%d/%d | r=%d | residual-final | nTrain=%d\n', ...
    numel(data.trainCombos), numel(data.valCombos), numel(data.testCombos), ...
    data.historyLen, numel(data.XTrain));

model = trainLSTM(data, 'modelType', 'lstm', 'maxEpochs', opt.maxEpochs, ...
    'hiddenUnits', opt.hiddenUnits, 'ProjectRoot', proj, 'miniBatchSize', 16);
model = rename_model_file(model, proj, sprintf('lstm_resid_h%d', data.historyLen));

% Evaluate in absolute FA/FB space
[mLSTM, Yt, Yp] = eval_final_absolute(model, data, 'test');
mPers = eval_persistence_final(data);
mLin = eval_linear_final(data);

% Also absolute-target LSTM for comparison
dataAbs = prepareSequenceData_final(samples, 'historyLen', opt.historyLen, 'residual', false);
modelAbs = trainLSTM(dataAbs, 'modelType', 'lstm', 'maxEpochs', opt.maxEpochs, ...
    'hiddenUnits', opt.hiddenUnits, 'ProjectRoot', proj);
modelAbs = rename_model_file(modelAbs, proj, sprintf('lstm_abs_h%d', dataAbs.historyLen));
[mLSTM_abs, ~, ~] = eval_final_absolute(modelAbs, dataAbs, 'test');

tabDir = fullfile(proj, '06_Results', 'tables');
figDir = fullfile(proj, '06_Results', 'figures');
logDir = fullfile(proj, '06_Results', 'logs');
if ~exist(tabDir, 'dir'), mkdir(tabDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(logDir, 'dir'), mkdir(logDir); end

tab = fullfile(tabDir, 'stage4_lstm_improved.csv');
fid = fopen(tab, 'w');
fprintf(fid, 'model,MAE_finalA,RMSE_finalA,R2_finalA,MAE_finalB,R2_pair,n\n');
write_row(fid, 'persistence', mPers);
write_row(fid, 'linear', mLin);
write_row(fid, 'lstm_residual', mLSTM);
write_row(fid, 'lstm_absolute', mLSTM_abs);
fclose(fid);

figure('Visible', 'off');
scatter(Yt(1, :), Yp(1, :), 36, 'filled'); hold on;
lims = [min([Yt(1, :), Yp(1, :)]), max([Yt(1, :), Yp(1, :)])];
plot(lims, lims, 'k--');
xlabel('True F_A / final \sigma_A'); ylabel('Predicted');
title(sprintf('Improved LSTM residual (R^2_A=%.3f)', mLSTM.R2_A));
grid on; axis equal tight;
fig = fullfile(figDir, 'stage4_lstm_improved_FA.png');
saveas(gcf, fig); close(gcf);

result = struct('persistence', mPers, 'linear', mLin, ...
    'lstm_residual', mLSTM, 'lstm_absolute', mLSTM_abs, ...
    'opt', opt, 'files', struct('table', tab, 'fig', fig));
outMat = fullfile(logDir, 'stage4_lstm_improved.mat');
save(outMat, 'result', 'model', 'data', 'Yt', 'Yp');
result.files.mat = outMat;

fprintf('\n=== Improved FA prediction ===\n');
fprintf('persistence:  MAE_A=%.2f  R2_A=%.3f\n', mPers.MAE_A, mPers.R2_A);
fprintf('linear:       MAE_A=%.2f  R2_A=%.3f\n', mLin.MAE_A, mLin.R2_A);
fprintf('lstm_abs:     MAE_A=%.2f  R2_A=%.3f\n', mLSTM_abs.MAE_A, mLSTM_abs.R2_A);
fprintf('lstm_resid:   MAE_A=%.2f  R2_A=%.3f  MAE_B=%.2f  R2_pair=%.3f\n', ...
    mLSTM.MAE_A, mLSTM.R2_A, mLSTM.MAE_B, mLSTM.R2_pair);
fprintf('Saved %s\n', outMat);
end

function write_row(fid, name, m)
fprintf(fid, '%s,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n', name, ...
    m.MAE_A, m.RMSE_A, m.R2_A, m.MAE_B, m.R2_pair, m.n);
end

function [metrics, Yt, Yp] = eval_final_absolute(model, data, splitName)
switch lower(splitName)
    case 'train'
        Xc = data.XTrain; Ym = data.YTrain; base = data.baseTrain;
    case 'val'
        Xc = data.XVal; Ym = data.YVal; base = data.baseVal;
    otherwise
        Xc = data.XTest; Ym = data.YTest; base = data.baseTest;
end
n = numel(Xc);
Yt = zeros(2, n); Yp = zeros(2, n);
for i = 1:n
    yhat = predict(model.net, Xc{i});
    yhat = yhat(:) .* model.sigY + model.muY;
    ytrue = Ym(i, :)' .* data.sigY + data.muY;
    if data.residual
        yhat = yhat + base(i, :)';
        ytrue = ytrue + base(i, :)';
    end
    yhat = max(yhat, 0);
    Yp(:, i) = yhat;
    Yt(:, i) = ytrue;
end
metrics = score_pair(Yt, Yp);
end

function metrics = eval_persistence_final(data)
n = numel(data.XTest);
Yt = zeros(2, n); Yp = zeros(2, n);
for i = 1:n
    base = data.baseTest(i, :)';
    ytrue = data.YTest(i, :)' .* data.sigY + data.muY;
    if data.residual
        ytrue = ytrue + base;
    end
    Yp(:, i) = max(base, 0);
    Yt(:, i) = ytrue;
end
metrics = score_pair(Yt, Yp);
end

function metrics = eval_linear_final(data)
nTr = numel(data.XTrain); nTe = numel(data.XTest);
d = size(data.XTrain{1}, 1);
F = ones(nTr, d+1); Y = zeros(nTr, 2);
for i = 1:nTr
    F(i, 2:end) = data.XTrain{i}(:, end)';
    y = data.YTrain(i, :)' .* data.sigY + data.muY;
    if data.residual
        y = y + data.baseTrain(i, :)';
    end
    Y(i, :) = y';
end
beta = F \ Y;
Ft = ones(nTe, d+1);
Yt = zeros(2, nTe); Yp = zeros(2, nTe);
for i = 1:nTe
    Ft(i, 2:end) = data.XTest{i}(:, end)';
    ytrue = data.YTest(i, :)' .* data.sigY + data.muY;
    if data.residual
        ytrue = ytrue + data.baseTest(i, :)';
    end
    Yt(:, i) = ytrue;
end
Yhat = Ft * beta;
Yp = max(Yhat', 0);
metrics = score_pair(Yt, Yp);
end

function m = score_pair(Yt, Yp)
err = Yp - Yt;
m = struct();
m.n = size(Yt, 2);
m.MAE_A = mean(abs(err(1, :)));
m.RMSE_A = sqrt(mean(err(1, :).^2));
m.R2_A = 1 - sum(err(1, :).^2) / max(sum((Yt(1, :) - mean(Yt(1, :))).^2), eps);
m.MAE_B = mean(abs(err(2, :)));
m.RMSE_B = sqrt(mean(err(2, :).^2));
m.R2_B = 1 - sum(err(2, :).^2) / max(sum((Yt(2, :) - mean(Yt(2, :))).^2), eps);
m.R2_pair = 1 - sum(err(:).^2) / max(sum((Yt(:) - mean(Yt(:))).^2), eps);
m.MAE = mean(abs(err), 'all');
end

function model = rename_model_file(model, proj, tag)
outFile = fullfile(proj, '03_LSTM', 'models', sprintf('model_%s.mat', tag));
save(outFile, 'model');
model.file = outFile;
end

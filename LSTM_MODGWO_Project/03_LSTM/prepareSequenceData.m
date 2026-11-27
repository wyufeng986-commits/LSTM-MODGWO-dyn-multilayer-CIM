function data = prepareSequenceData(samples, varargin)
%PREPARESEQUENCEDATA  Leakage-safe split by seed combo + history/future windows.

parser = inputParser;
parser.addParameter('historyLen', 8, @isscalar);
parser.addParameter('trainRatio', 0.7, @isscalar);
parser.addParameter('valRatio', 0.15, @isscalar);
parser.addParameter('rngSeed', 1, @isscalar);
parser.parse(varargin{:});
opt = parser.Results;

records = samples.records;
T = samples.T;
r = min(opt.historyLen, T - 1);
horizon = T - r;

comboIds = unique([records.comboId]);
rng(opt.rngSeed);
comboIds = comboIds(randperm(numel(comboIds)));
n = numel(comboIds);
nTrain = max(1, floor(opt.trainRatio * n));
nVal = max(1, floor(opt.valRatio * n));
if nTrain + nVal >= n
    nVal = max(1, n - nTrain - 1);
end
nTest = n - nTrain - nVal;

trainCombos = comboIds(1:nTrain);
valCombos = comboIds(nTrain+1:nTrain+nVal);
testCombos = comboIds(nTrain+nVal+1:end);

[XTrain, YTrainMat] = pack_split(records, trainCombos, r, T);
[XVal, YValMat] = pack_split(records, valCombos, r, T);
[XTest, YTestMat] = pack_split(records, testCombos, r, T);

% normalization from training set only
[XTrain, YTrainMat, muX, sigX, muY, sigY] = normalize_xy(XTrain, YTrainMat, [], [], [], []);
[XVal, YValMat] = normalize_xy(XVal, YValMat, muX, sigX, muY, sigY);
[XTest, YTestMat] = normalize_xy(XTest, YTestMat, muX, sigX, muY, sigY);

data = struct();
data.historyLen = r;
data.horizon = horizon;
data.T = T;
data.XTrain = XTrain; data.YTrain = YTrainMat; % Y: nObs x (2*H)
data.XVal = XVal; data.YVal = YValMat;
data.XTest = XTest; data.YTest = YTestMat;
data.muX = muX; data.sigX = sigX;
data.muY = muY; data.sigY = sigY;
data.trainCombos = trainCombos;
data.valCombos = valCombos;
data.testCombos = testCombos;
data.featureDim = size(XTrain{1}, 1);
data.responseDim = size(YTrainMat, 2);
end

function [Xcell, Ymat] = pack_split(records, combos, r, T)
Xcell = {};
Yrows = {};
k = 0;
for i = 1:numel(records)
    if ~ismember(records(i).comboId, combos), continue; end
    X = records(i).X; % F x T
    Yfull = records(i).Y; % 2 x T
    Xin = X(:, 1:r);
    Yout = Yfull(:, r+1:T); % 2 x horizon
    k = k + 1;
    Xcell{k, 1} = Xin; %#ok<AGROW>
    Yrows{k, 1} = Yout(:)'; % 1 x (2*horizon), column-major of 2xH
end
if k == 0
    Ymat = zeros(0, 2 * (T - r));
else
    Ymat = cell2mat(Yrows);
end
end

function [Xcell, Ymat, muX, sigX, muY, sigY] = normalize_xy(Xcell, Ymat, muX, sigX, muY, sigY)
if isempty(Xcell)
    return;
end
if isempty(muX)
    allX = cat(2, Xcell{:});
    muX = mean(allX, 2);
    sigX = std(allX, 0, 2);
    sigX(sigX < 1e-8) = 1;
    muY = mean(Ymat, 1)';
    sigY = std(Ymat, 0, 1)';
    sigY(sigY < 1e-8) = 1;
end
for i = 1:numel(Xcell)
    Xcell{i} = (Xcell{i} - muX) ./ sigX;
end
Ymat = (Ymat - muY') ./ sigY';
end

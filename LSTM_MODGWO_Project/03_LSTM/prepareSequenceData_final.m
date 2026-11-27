function data = prepareSequenceData_final(samples, varargin)
%PREPARESEQUENCEDATA_FINAL  Predict only final (sigmaA, sigmaB), optionally residual.
%   This matches optimization needs (approximate F_A) better than full multi-step.

parser = inputParser;
parser.addParameter('historyLen', 4, @isscalar);
parser.addParameter('trainRatio', 0.7, @isscalar);
parser.addParameter('valRatio', 0.15, @isscalar);
parser.addParameter('rngSeed', 1, @isscalar);
parser.addParameter('residual', true, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
opt = parser.Results;

records = samples.records;
T = samples.T;
r = min(opt.historyLen, T - 1);

comboIds = unique([records.comboId]);
rng(opt.rngSeed);
comboIds = comboIds(randperm(numel(comboIds)));
n = numel(comboIds);
nTrain = max(1, floor(opt.trainRatio * n));
nVal = max(1, floor(opt.valRatio * n));
if nTrain + nVal >= n
    nVal = max(1, n - nTrain - 1);
end

trainCombos = comboIds(1:nTrain);
valCombos = comboIds(nTrain+1:nTrain+nVal);
testCombos = comboIds(nTrain+nVal+1:end);

[XTrain, YTrain, baseTrain] = pack_split(records, trainCombos, r, T, opt.residual);
[XVal, YVal, baseVal] = pack_split(records, valCombos, r, T, opt.residual);
[XTest, YTest, baseTest] = pack_split(records, testCombos, r, T, opt.residual);

[XTrain, YTrain, muX, sigX, muY, sigY] = normalize_xy(XTrain, YTrain, [], [], [], []);
[XVal, YVal] = normalize_xy(XVal, YVal, muX, sigX, muY, sigY);
[XTest, YTest] = normalize_xy(XTest, YTest, muX, sigX, muY, sigY);

data = struct();
data.historyLen = r;
data.horizon = 1;
data.T = T;
data.residual = logical(opt.residual);
data.XTrain = XTrain; data.YTrain = YTrain;
data.XVal = XVal; data.YVal = YVal;
data.XTest = XTest; data.YTest = YTest;
data.baseTrain = baseTrain; data.baseVal = baseVal; data.baseTest = baseTest;
data.muX = muX; data.sigX = sigX;
data.muY = muY; data.sigY = sigY;
data.trainCombos = trainCombos;
data.valCombos = valCombos;
data.testCombos = testCombos;
data.featureDim = size(XTrain{1}, 1);
data.responseDim = size(YTrain, 2);
end

function [Xcell, Ymat, baseMat] = pack_split(records, combos, r, T, residual)
Xcell = {};
Yrows = {};
Brows = {};
k = 0;
for i = 1:numel(records)
    if ~ismember(records(i).comboId, combos), continue; end
    X = records(i).X;
    Yfull = records(i).Y; % 2 x T
    Xin = X(:, 1:r);
    finalAB = Yfull(:, T)'; % 1 x 2
    baseAB = Yfull(:, r)';  % last observed in history
    if residual
        target = finalAB - baseAB;
    else
        target = finalAB;
    end
    k = k + 1;
    Xcell{k, 1} = Xin; %#ok<AGROW>
    Yrows{k, 1} = target;
    Brows{k, 1} = baseAB;
end
if k == 0
    Ymat = zeros(0, 2);
    baseMat = zeros(0, 2);
else
    Ymat = cell2mat(Yrows);
    baseMat = cell2mat(Brows);
end
end

function [Xcell, Ymat, muX, sigX, muY, sigY] = normalize_xy(Xcell, Ymat, muX, sigX, muY, sigY)
if isempty(Xcell), return; end
if isempty(muX)
    allX = cat(2, Xcell{:});
    muX = mean(allX, 2);
    sigX = std(allX, 0, 2); sigX(sigX < 1e-8) = 1;
    muY = mean(Ymat, 1)';
    sigY = std(Ymat, 0, 1)'; sigY(sigY < 1e-8) = 1;
end
for i = 1:numel(Xcell)
    Xcell{i} = (Xcell{i} - muX) ./ sigX;
end
Ymat = (Ymat - muY') ./ sigY';
end

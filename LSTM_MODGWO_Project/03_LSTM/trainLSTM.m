function model = trainLSTM(data, varargin)
%TRAINLSTM  Train multi-step dual-spread sequence regressor (LSTM/GRU).

parser = inputParser;
parser.addParameter('hiddenUnits', 64, @isscalar);
parser.addParameter('maxEpochs', 40, @isscalar);
parser.addParameter('miniBatchSize', 16, @isscalar);
parser.addParameter('modelType', 'lstm', @(x) ischar(x) || isstring(x));
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opt = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opt.ProjectRoot)) == 0
    opt.ProjectRoot = fileparts(thisDir);
end
proj = char(opt.ProjectRoot);
outDir = fullfile(proj, '03_LSTM', 'models');
if ~exist(outDir, 'dir'); mkdir(outDir); end

numF = data.featureDim;
numOut = data.responseDim;
mtype = lower(string(opt.modelType));

if mtype == "gru"
    core = gruLayer(opt.hiddenUnits, 'OutputMode', 'last', 'Name', 'rnn');
else
    core = lstmLayer(opt.hiddenUnits, 'OutputMode', 'last', 'Name', 'rnn');
    mtype = "lstm";
end

layers = [ ...
    sequenceInputLayer(numF, 'Name', 'in')
    core
    fullyConnectedLayer(numOut, 'Name', 'fc')
    regressionLayer('Name', 'reg')];

options = trainingOptions('adam', ...
    'MaxEpochs', opt.maxEpochs, ...
    'MiniBatchSize', opt.miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {data.XVal, data.YVal}, ...
    'ValidationFrequency', max(1, floor(numel(data.XTrain) / opt.miniBatchSize)), ...
    'InitialLearnRate', 1e-3, ...
    'GradientThreshold', 1, ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'Plots', 'none');

fprintf('Training %s: features=%d out=%d train=%d val=%d\n', ...
    mtype, numF, numOut, numel(data.XTrain), numel(data.XVal));
net = trainNetwork(data.XTrain, data.YTrain, layers, options);

model = struct();
model.net = net;
model.type = char(mtype);
model.historyLen = data.historyLen;
model.horizon = data.horizon;
model.muX = data.muX;
model.sigX = data.sigX;
model.muY = data.muY;
model.sigY = data.sigY;
model.featureDim = numF;
model.responseDim = numOut;
model.opt = opt;

tag = sprintf('%s_h%d', mtype, data.historyLen);
outFile = fullfile(outDir, sprintf('model_%s.mat', tag));
save(outFile, 'model');
model.file = outFile;
fprintf('Saved %s\n', outFile);
end

function samples = generateLSTMSamples(varargin)
%GENERATELSTMSAMPLES  Build competitive-spread trajectory samples for LSTM.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('nSeedCombos', 40, @isscalar);
parser.addParameter('nTrajPerCombo', 3, @isscalar);
parser.addParameter('kA', 5, @isscalar);
parser.addParameter('kB', 5, @isscalar);
parser.addParameter('beta', 1.0, @(x) isnumeric(x));
parser.addParameter('rngSeed', 20261127, @isscalar);
parser.addParameter('historyLen', 8, @isscalar);
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

Network = buildMultilayerNetwork('ProjectRoot', proj);
N = Network.numNodes;
T = Network.numSnapshots;
costInfo = calculateNodeCostC4(Network, struct('includeFacebook', false));
zG = extractNetworkFeatures(Network, struct('includeFacebook', false));

[SAlist, SB] = sampleRandomSeedSets(N, opt.kA, opt.kB, opt.nSeedCombos, costInfo.sOut, opt.rngSeed);

icOpts = struct('beta', opt.beta, 'includeFacebook', false, ...
    'maxRoundsPerSnapshot', 30, 'recordEveryRound', false);
icOpts.Pcache = prepareICProbabilities(Network, icOpts);

records = [];
recId = 0;
fprintf('Generating LSTM samples: combos=%d traj/combo=%d T=%d\n', ...
    opt.nSeedCombos, opt.nTrajPerCombo, T);

for c = 1:opt.nSeedCombos
    SA = SAlist{c};
    zS = extractSeedFeatures(Network, SA, SB, costInfo, struct('includeFacebook', false));
    for tr = 1:opt.nTrajPerCombo
        rng(opt.rngSeed + 10000 * c + tr);
        out = competitiveIC(Network, SA, SB, icOpts);
        sigA = out.history.sigmaA(:)';
        sigB = out.history.sigmaB(:)';
        % pad/truncate to T
        if numel(sigA) < T
            sigA = [sigA, repmat(sigA(end), 1, T - numel(sigA))];
            sigB = [sigB, repmat(sigB(end), 1, T - numel(sigB))];
        else
            sigA = sigA(1:T); sigB = sigB(1:T);
        end
        Xdyn = extractTemporalFeatures(sigA, sigB, N); % 7 x T
        hS = [zS; zG]; % static vector
        % broadcast static features on each step
        X = [Xdyn; repmat(hS, 1, T)]; % F x T
        Y = [sigA; sigB]; % 2 x T

        recId = recId + 1;
        records(recId).comboId = c; %#ok<AGROW>
        records(recId).trajId = tr;
        records(recId).S_A = SA;
        records(recId).S_B = SB;
        records(recId).X = X;
        records(recId).Y = Y;
        records(recId).finalA = sigA(end);
        records(recId).finalB = sigB(end);
    end
    if mod(c, 5) == 0 || c == opt.nSeedCombos
        fprintf('  combo %d/%d done\n', c, opt.nSeedCombos);
    end
end

samples = struct();
samples.records = records;
samples.opt = opt;
samples.N = N;
samples.T = T;
samples.SB = SB;
samples.featureDim = size(records(1).X, 1);
samples.zG = zG;
samples.costInfo = costInfo;
samples.note = "Labels from competitive IC simulation; not observational CNS infection labels.";

outDir = fullfile(proj, '03_LSTM', 'models');
if ~exist(outDir, 'dir'); mkdir(outDir); end
outMat = fullfile(outDir, 'lstm_raw_samples.mat');
save(outMat, 'samples', '-v7.3');
samples.file = outMat;
fprintf('Saved %s (%d trajectories)\n', outMat, numel(records));
end

function run_propagation_smoke()
%RUN_PROPAGATION_SMOKE  Stage-2 verification on CNS daily multilayer network.

proj = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(proj, '01_Data'));
addpath(fullfile(proj, '02_Propagation'));

Network = buildMultilayerNetwork('ProjectRoot', proj);
N = Network.numNodes;

% Aggregate out-strength over main dynamic layers / all snapshots
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
[~, ord] = sort(outStr, 'descend');

kA = 5;
kB = 5;
S_B = ord(1:kB)';
remain = ord(kB+1:end);
S_A = remain(1:kA)';

assert(isempty(intersect(S_A, S_B)), 'seed overlap');

opts = struct();
opts.beta = 1.0;
opts.R = 10;
opts.rngSeed = 20261127;
opts.maxRoundsPerSnapshot = 30;
opts.includeFacebook = false;

fprintf('Smoke seeds S_A=%s\n', mat2str(S_A));
fprintf('Smoke seeds S_B=%s\n', mat2str(S_B));
fprintf('Layers: %s\n', strjoin(string({Network.layer(layerIdx).name}), ', '));

% Reproducibility: identical seed -> identical single-run trajectory encoding
rng(1);
r1 = competitiveIC(Network, S_A, S_B, opts);
rng(1);
r2 = competitiveIC(Network, S_A, S_B, opts);
assert(isequal(r1.state, r2.state), 'reproducibility failed');
assert(isempty(intersect(r1.I_A, r1.I_B)), 'node claimed by both sides');
assert(all(r1.state(S_A) == 1) && all(r1.state(S_B) == 2), 'seeds corrupted');

mc = runMonteCarlo(Network, S_A, S_B, opts);
fprintf('Single-run: sigmaA=%d sigmaB=%d steps=%d\n', r1.sigmaA, r1.sigmaB, numel(r1.history.sigmaA));
fprintf('MC R=%d: F_A=%.3f (+/-%.3f), F_B=%.3f (+/-%.3f)\n', ...
    mc.R, mc.F_A, mc.std_A, mc.F_B, mc.std_B);

outFig = fullfile(proj, '06_Results', 'figures', 'stage2_propagation_mc.png');
visualizePropagation(mc, outFig);
outFig1 = fullfile(proj, '06_Results', 'figures', 'stage2_propagation_single.png');
visualizePropagation(r1, outFig1);

outMat = fullfile(proj, '06_Results', 'logs', 'stage2_smoke_result.mat');
if ~exist(fileparts(outMat), 'dir'); mkdir(fileparts(outMat)); end
save(outMat, 'Network', 'S_A', 'S_B', 'r1', 'mc', 'opts');
fprintf('Saved %s\n', outMat);
fprintf('Saved %s\n', outFig);
fprintf('STAGE2_SMOKE_OK\n');
end

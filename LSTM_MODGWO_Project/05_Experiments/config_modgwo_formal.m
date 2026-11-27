function cfg = config_modgwo_formal(preset)
%CONFIG_MODGWO_FORMAL  Formal Stage-5 MODGWO presets (do not run heavy jobs here).
%
% preset:
%   'formal'     - default paper-scale hybrid (k=5, matches residual LSTM)
%   'formal_mc'  - same scale, pure MC (no LSTM); slower, for ablation later
%   'medium'     - intermediate scale for first remote dry-run
%   'smoke'      - tiny local sanity (already used)

if nargin < 1 || isempty(preset)
    preset = 'formal';
end
preset = lower(string(preset));

cfg = struct();
cfg.preset = char(preset);
cfg.evalMode = 'hybrid';
cfg.kA = 5;
cfg.kB = 5;
cfg.beta = 1.0;
cfg.rngSeed = 20261127;
cfg.finalCorrect = true;
cfg.localSearchTries = 3;
cfg.localSearchProb = 0.35;
cfg.leaderBeta = 4;
cfg.lstmModelRel = fullfile('03_LSTM', 'models', 'model_lstm_resid_h4.mat');
cfg.note = '';

switch preset
    case "smoke"
        cfg.popSize = 8;
        cfg.maxIter = 4;
        cfg.archiveSize = 15;
        cfg.R_fast = 2;
        cfg.R_final = 5;
        cfg.correctEvery = 2;
        cfg.note = 'Tiny sanity preset.';

    case "medium"
        % First remote dry-run before full formal
        cfg.popSize = 20;
        cfg.maxIter = 20;
        cfg.archiveSize = 40;
        cfg.R_fast = 4;
        cfg.R_final = 20;
        cfg.correctEvery = 4;
        cfg.note = 'Intermediate remote dry-run.';

    case "formal_mc"
        cfg.evalMode = 'mc';
        cfg.popSize = 50;
        cfg.maxIter = 50;
        cfg.archiveSize = 100;
        cfg.R_fast = 10;
        cfg.R_final = 50;
        cfg.correctEvery = 0;
        cfg.localSearchProb = 0.25;
        cfg.note = 'Pure-MC ablation-scale run; expect much longer wall time.';

    case "formal"
        cfg.popSize = 50;
        cfg.maxIter = 50;
        cfg.archiveSize = 100;
        cfg.R_fast = 5;       % periodic archive MC under hybrid
        cfg.R_final = 50;     % final archive correction
        cfg.correctEvery = 5;
        cfg.note = [ ...
            'Default formal hybrid. kA=kB=5 matches residual LSTM training. ', ...
            'Do NOT use k~0.05N with current LSTM without retraining.'];

    otherwise
        error('config_modgwo_formal:Preset', 'Unknown preset: %s', preset);
end

% Rough complexity hint (not a promise)
cfg.estPopIters = cfg.popSize * cfg.maxIter;
cfg.tag = sprintf('%s_p%d_i%d_k%d_%s', cfg.preset, cfg.popSize, ...
    cfg.maxIter, cfg.kA, cfg.evalMode);
end

%RUN_FORMAL_MODGWO  One-click formal Stage-5 entry for a remote machine.
%
% Usage (MATLAB Command Window):
%   cd('<path-to>/LSTM_MODGWO_Project')
%   run_formal_modgwo
%
% Or batch:
%   matlab -batch "cd('<path-to>/LSTM_MODGWO_Project'); run_formal_modgwo"
%
% Change PRESET below if needed: 'medium' | 'formal' | 'formal_mc'

PRESET = 'formal';   % recommended first remote try: 'medium'

thisDir = fileparts(mfilename('fullpath'));
cd(thisDir);
addpath(genpath(thisDir));

fprintf('run_formal_modgwo: preset=%s\n', PRESET);
cfg = config_modgwo_formal(PRESET);
disp(cfg);

result = experimentMODGWO_formal('ProjectRoot', thisDir, 'preset', PRESET);

fprintf('\nDONE runId=%s\n', result.runId);
fprintf('Copy back at least:\n  %s\n  %s\n  %s\n', ...
    result.files.mat, result.files.archiveCsv, result.files.diary);

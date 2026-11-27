function preprocessCNS(varargin)
%PREPROCESSCNS  Call Python Stage-1 builder to produce CNS_daily_multilayer.mat.
% Requires: Python with pandas + scipy.
% Prefer running preprocess_cns_python.py directly if MATLAB cannot find Python.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opts = parser.Results;
thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opts.ProjectRoot)) == 0
    opts.ProjectRoot = fileparts(thisDir);
end
pyScript = fullfile(thisDir, 'preprocess_cns_python.py');
if ~isfile(pyScript)
    error('preprocessCNS:MissingScript', 'Missing %s', pyScript);
end

cmd = sprintf('python "%s"', pyScript);
fprintf('Running: %s\n', cmd);
[status, cmdout] = system(cmd);
fprintf('%s\n', cmdout);
if status ~= 0
    error('preprocessCNS:PythonFailed', 'Python preprocess failed with status %d', status);
end
end

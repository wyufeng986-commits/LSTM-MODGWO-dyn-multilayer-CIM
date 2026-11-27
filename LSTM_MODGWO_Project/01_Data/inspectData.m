function report = inspectData(varargin)
%INSPECTDATA  Inspect extracted CNS layer CSVs and write a JSON/text report.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opts = parser.Results;
thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opts.ProjectRoot)) == 0
    opts.ProjectRoot = fileparts(thisDir);
end
rawDir = fullfile(char(opts.ProjectRoot), '01_Data', 'raw', 'cns');
outDir = fullfile(char(opts.ProjectRoot), '01_Data', 'processed');
if ~exist(outDir, 'dir'); mkdir(outDir); end

layers = {'calls', 'sms', 'bt', 'fb_friends'};
report = struct();
report.sourceDir = rawDir;
report.layers = struct();

for i = 1:numel(layers)
    name = layers{i};
    edgesFile = fullfile(rawDir, name, 'edges.csv');
    nodesFile = fullfile(rawDir, name, 'nodes.csv');
    info = struct('available', false);
    if isfile(edgesFile) && isfile(nodesFile)
        edges = readtable(edgesFile, 'FileType', 'text');
        nodes = readtable(nodesFile, 'FileType', 'text');
        info.available = true;
        info.nEdges = height(edges);
        info.nNodes = height(nodes);
        info.edgeVars = edges.Properties.VariableNames;
        info.nodeVars = nodes.Properties.VariableNames;
    end
    report.layers.(name) = info;
    fprintf('[%s] available=%d\n', name, info.available);
end

save(fullfile(outDir, 'inspect_report_matlab.mat'), 'report');
fprintf('Saved %s\n', fullfile(outDir, 'inspect_report_matlab.mat'));
end

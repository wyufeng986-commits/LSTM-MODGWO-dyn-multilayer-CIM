function Network = buildMultilayerNetwork(varargin)
%BUILDMULTILAYERNETWORK  Load processed CNS Network MAT and normalize layout.
% Python scipy.io.savemat stores layer as 1xL cell of structs; A/W may be dense.
% This loader converts to MATLAB-friendly struct array + sparse snapshots.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.addParameter('Rebuild', false, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
opts = parser.Results;
thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opts.ProjectRoot)) == 0
    opts.ProjectRoot = fileparts(thisDir);
end
matPath = fullfile(char(opts.ProjectRoot), '01_Data', 'processed', 'CNS_daily_multilayer.mat');
if opts.Rebuild || ~isfile(matPath)
    preprocessCNS('ProjectRoot', opts.ProjectRoot);
end
S = load(matPath, 'Network');
Network = normalize_network_layout(S.Network);
fprintf('Loaded Network: N=%g L=%g T=%g\n', Network.numNodes, Network.numLayers, Network.numSnapshots);
end

function Network = normalize_network_layout(Network)
if iscell(Network.layer)
    L = numel(Network.layer);
    tmp = Network.layer{1};
    layers = repmat(tmp, 1, L);
    for l = 1:L
        layers(l) = normalize_layer(Network.layer{l});
    end
    Network.layer = layers;
else
    for l = 1:numel(Network.layer)
        Network.layer(l) = normalize_layer(Network.layer(l));
    end
end
if iscell(Network.layerNames)
    % keep as cellstr
elseif isstring(Network.layerNames)
    Network.layerNames = cellstr(Network.layerNames);
end
end

function layer = normalize_layer(layer)
T = numel(layer.A);
A = cell(1, T);
W = cell(1, T);
E = cell(1, T);
for t = 1:T
    At = layer.A{t};
    Wt = layer.W{t};
    if ~issparse(At); At = sparse(At); end
    if ~issparse(Wt); Wt = sparse(Wt); end
    A{t} = At;
    W{t} = Wt;
    % Always rebuild edgeList from W to avoid scipy/MATLAB cell quirks.
    [ii, jj, vv] = find(Wt);
    if isempty(ii)
        E{t} = zeros(0, 3);
    else
        E{t} = [ii, jj, vv];
    end
end
layer.A = A;
layer.W = W;
layer.edgeList = E;
if ~isfield(layer, 'numSnapshots') || isempty(layer.numSnapshots)
    layer.numSnapshots = T;
end
end

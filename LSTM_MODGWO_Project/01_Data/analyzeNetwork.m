function stats = analyzeNetwork(varargin)
%ANALYZENETWORK  Basic stats/tables for processed CNS Network.

parser = inputParser;
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opts = parser.Results;
thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opts.ProjectRoot)) == 0
    opts.ProjectRoot = fileparts(thisDir);
end
Network = buildMultilayerNetwork('ProjectRoot', opts.ProjectRoot);

outDir = fullfile(char(opts.ProjectRoot), '06_Results', 'tables');
figDir = fullfile(char(opts.ProjectRoot), '06_Results', 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end
if ~exist(figDir, 'dir'); mkdir(figDir); end

L = Network.numLayers;
T = Network.numSnapshots;
stats = struct();
stats.numNodes = Network.numNodes;
stats.numLayers = L;
stats.numSnapshots = T;

fid = fopen(fullfile(outDir, 'network_layer_stats.csv'), 'w');
fprintf(fid, 'layer,role,dynamic,mean_edges,max_edges,nonempty_days,mean_density\n');

figure('Visible', 'off'); hold on;
for l = 1:L
    layer = Network.layer(l);
    counts = zeros(1, T);
    for t = 1:T
        E = layer.edgeList{t};
        counts(t) = size(E, 1);
    end
    dens = counts ./ max(Network.numNodes * (Network.numNodes - 1), 1);
    if iscell(Network.layerNames)
        name = char(Network.layerNames{l});
    else
        name = char(layer.name);
    end
    role = char(layer.role);
    fprintf(fid, '%s,%s,%d,%.3f,%d,%d,%.8f\n', name, role, logical(layer.dynamic), ...
        mean(counts), max(counts), nnz(counts), mean(dens));
    stats.(matlab.lang.makeValidName(name)).edges_per_day = counts;
    if logical(layer.dynamic)
        plot(1:T, counts, '-o', 'DisplayName', name);
    end
end
fclose(fid);
xlabel('Day'); ylabel('Directed arcs'); title('CNS daily edge counts');
legend('Location', 'best'); grid on;
saveas(gcf, fullfile(figDir, 'cns_daily_edge_counts_matlab.png'));
close(gcf);

fprintf('Wrote %s\n', fullfile(outDir, 'network_layer_stats.csv'));
end

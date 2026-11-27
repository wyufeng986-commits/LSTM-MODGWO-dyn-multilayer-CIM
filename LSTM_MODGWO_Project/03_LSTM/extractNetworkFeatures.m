function zG = extractNetworkFeatures(Network, opts)
%EXTRACTNETWORKFEATURES  Static network features (shared across samples).

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'includeFacebook'), opts.includeFacebook = false; end

layerIdx = selectPropagationLayers(Network, opts);
N = Network.numNodes;
T = Network.numSnapshots;
L = numel(layerIdx);

edgeCounts = zeros(1, L);
meanW = zeros(1, L);
meanOut = zeros(1, L);
dens = zeros(1, L);
for li = 1:L
    l = layerIdx(li);
    ec = 0; ws = 0; outs = zeros(N, 1);
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W), W = sparse(W); end
        ec = ec + nnz(W);
        ws = ws + full(sum(nonzeros(W)));
        outs = outs + full(sum(W, 2));
    end
    edgeCounts(li) = ec / T;
    if ec > 0
        meanW(li) = ws / max(ec, 1);
    end
    meanOut(li) = mean(outs / T);
    dens(li) = edgeCounts(li) / max(N * (N - 1), 1);
end

% dynamic change rate: mean Jaccard distance of edge supports across consecutive days
chg = zeros(1, L);
for li = 1:L
    l = layerIdx(li);
    acc = 0; cnt = 0;
    prev = [];
    for t = 1:T
        W = Network.layer(l).W{t};
        if ~issparse(W), W = sparse(W); end
        cur = find(W > 0);
        if t > 1
            inter = numel(intersect(prev, cur));
            uni = numel(union(prev, cur));
            if uni > 0
                acc = acc + (1 - inter / uni);
                cnt = cnt + 1;
            end
        end
        prev = cur;
    end
    if cnt > 0, chg(li) = acc / cnt; end
end

zG = [ ...
    N; L; T; ...
    mean(edgeCounts); sum(edgeCounts); ...
    mean(meanW); mean(meanOut); mean(dens); mean(chg) ...
    ];
zG = zG(:);
end

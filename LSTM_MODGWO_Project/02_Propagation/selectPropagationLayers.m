function layerIdx = selectPropagationLayers(Network, opts)
%SELECTPROPAGATIONLAYERS  Choose layers for IC (default: main dynamic only).

if nargin < 2
    opts = struct();
end
L = Network.numLayers;
names = string({Network.layer.name});

if isfield(opts, 'layerNames') && ~isempty(opts.layerNames)
    want = string(opts.layerNames);
    layerIdx = find(ismember(names, want));
elseif isfield(opts, 'includeFacebook') && opts.includeFacebook
    layerIdx = 1:L;
else
    % main dynamic layers; exclude static_ablation by default
    layerIdx = [];
    for l = 1:L
        role = string(Network.layer(l).role);
        dyn = logical(Network.layer(l).dynamic);
        if dyn && role ~= "static_ablation"
            layerIdx(end+1) = l; %#ok<AGROW>
        end
    end
    if isempty(layerIdx)
        layerIdx = 1:L;
    end
end
layerIdx = layerIdx(:)';
end

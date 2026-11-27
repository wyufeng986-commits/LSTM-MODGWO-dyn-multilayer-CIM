function obj = evaluateSeedObjectives(Network, S_A, S_B, costInfo, opts)
%EVALUATESEEDOBJECTIVES  Return F_A (max) and Cost_C4 (min) plus FB metric.
%   obj.F_A, obj.Cost, obj.F_B, obj.CostDetail

if nargin < 5 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'R') || isempty(opts.R), opts.R = 12; end
if ~isfield(opts, 'rngSeed') || isempty(opts.rngSeed), opts.rngSeed = 1; end

S_A = unique(S_A(:))';
if ~isempty(intersect(S_A, S_B(:)'))
    error('evaluateSeedObjectives:Overlap', 'S_A intersects S_B.');
end

mc = runMonteCarlo(Network, S_A, S_B, opts);
nodeCost = costInfo.nodeCost(:);
costVal = sum(nodeCost(S_A));

obj = struct();
obj.F_A = mc.F_A;
obj.F_B = mc.F_B;
obj.Cost = costVal;
obj.CostDetail = struct( ...
    'sOut', sum(costInfo.sOut(S_A)), ...
    'Lcount', sum(costInfo.Lcount(S_A)), ...
    'Dcount', sum(costInfo.Dcount(S_A)), ...
    'sHat', sum(costInfo.sHat(S_A)), ...
    'LHat', sum(costInfo.LHat(S_A)), ...
    'DHat', sum(costInfo.DHat(S_A)));
obj.S_A = S_A;
obj.mc = mc;
end

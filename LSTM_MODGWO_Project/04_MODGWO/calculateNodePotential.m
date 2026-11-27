function potential = calculateNodePotential(costInfo, S_B)
%CALCULATENODEPOTENTIAL  Deployment-efficiency potential H(v).
%   H(v) = sOut(v) / (Cost(v) + eps); S_B nodes forced to -inf.

sOut = costInfo.sOut(:);
c = costInfo.nodeCost(:);
potential = sOut ./ (c + 1e-8);
potential = potential(:)';
if nargin >= 2 && ~isempty(S_B)
    potential(S_B(:)) = -inf;
end
end

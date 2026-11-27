function [state, newlyA, newlyB] = updateNodeState(state, probA, probB, epsVal)
%UPDATENODESTATE  Competitive activation for inactive nodes.
%   state: N x 1, 0 inactive, 1 = A, 2 = B
%   probA/probB: N x 1 attempt-success probabilities under independent channels
%
%   Rule:
%     - sample successA ~ Bern(probA), successB ~ Bern(probB) for inactive nodes
%     - only A -> A; only B -> B
%     - both -> assign A with rho = probA/(probA+probB), else B

if nargin < 4 || isempty(epsVal)
    epsVal = 1e-12;
end

state = state(:);
probA = probA(:);
probB = probB(:);
N = numel(state);
inactive = (state == 0);

newlyA = false(N, 1);
newlyB = false(N, 1);
if ~any(inactive)
    return;
end

idx = find(inactive);
pA = probA(idx);
pB = probB(idx);

succA = rand(numel(idx), 1) < pA;
succB = rand(numel(idx), 1) < pB;

onlyA = succA & ~succB;
onlyB = succB & ~succA;
both = succA & succB;

if any(both)
    rho = pA(both) ./ (pA(both) + pB(both) + epsVal);
    takeA = rand(nnz(both), 1) < rho;
    bothIdx = idx(both);
    newlyA(bothIdx(takeA)) = true;
    newlyB(bothIdx(~takeA)) = true;
end

newlyA(idx(onlyA)) = true;
newlyB(idx(onlyB)) = true;

state(newlyA) = 1;
state(newlyB) = 2;
end

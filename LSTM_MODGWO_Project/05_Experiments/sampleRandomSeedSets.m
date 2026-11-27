function [SAlist, SB] = sampleRandomSeedSets(N, kA, kB, nSamples, outStr, rngSeed)
%SAMPLERANDOMSEEDSETS  Fix competitor seeds by out-strength; random feasible S_A.

if nargin < 6 || isempty(rngSeed)
    rngSeed = 20261127;
end
rng(rngSeed);

[~, ord] = sort(outStr(:), 'descend');
SB = ord(1:kB)';
pool = setdiff(1:N, SB, 'stable');
if numel(pool) < kA
    error('sampleRandomSeedSets:Pool', 'Not enough nodes for S_A.');
end

SAlist = cell(nSamples, 1);
for i = 1:nSamples
    pick = pool(randperm(numel(pool), kA));
    SAlist{i} = sort(pick);
end
end

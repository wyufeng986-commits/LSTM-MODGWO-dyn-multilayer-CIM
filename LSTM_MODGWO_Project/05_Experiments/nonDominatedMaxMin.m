function [mask, frontIdx] = nonDominatedMaxMin(f1, f2)
%NONDOMINATEDMAXMIN  Non-dominated filter for max f1, min f2.
%   i dominates j if f1(i)>=f1(j) and f2(i)<=f2(j), with at least one strict.

f1 = f1(:);
f2 = f2(:);
M = numel(f1);
mask = true(M, 1);
for i = 1:M
    for j = 1:M
        if i == j, continue; end
        if (f1(j) >= f1(i) && f2(j) <= f2(i)) && (f1(j) > f1(i) || f2(j) < f2(i))
            mask(i) = false;
            break;
        end
    end
end
frontIdx = find(mask);
end

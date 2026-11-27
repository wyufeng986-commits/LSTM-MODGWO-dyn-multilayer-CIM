function S = repairSeedSet(S, k, N, S_B, preference)
%REPAIRSEEDSET  Enforce unique length-k S_A disjoint from S_B.
%   preference: 1xN scores; higher preferred when filling / trimming.

S_B = unique(S_B(:)', 'stable');
S = unique(S(:)', 'stable');
S = setdiff(S, S_B, 'stable');
S = S(S >= 1 & S <= N);

if nargin < 5 || isempty(preference)
    preference = ones(1, N);
end
preference = preference(:)';

pool = setdiff(1:N, [S, S_B], 'stable');
if ~isempty(pool)
    [~, ord] = sort(preference(pool), 'descend');
    pool = pool(ord);
end

while numel(S) < k && ~isempty(pool)
    S(end + 1) = pool(1); %#ok<AGROW>
    pool(1) = [];
end

if numel(S) > k
    [~, ord] = sort(preference(S), 'descend');
    S = S(ord(1:k));
end

S = unique(S(:)', 'stable');
if numel(S) < k
    rest = setdiff(1:N, [S, S_B], 'stable');
    if ~isempty(rest)
        [~, ord] = sort(preference(rest), 'descend');
        rest = rest(ord);
        need = k - numel(S);
        S = [S, rest(1:min(need, numel(rest)))];
    end
end
S = sort(S(1:min(k, numel(S))));
end

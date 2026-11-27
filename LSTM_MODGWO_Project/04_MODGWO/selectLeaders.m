function leaders = selectLeaders(Archive, beta)
%SELECTLEADERS  Pick alpha/beta/delta via inverse grid-density roulette.
if nargin < 2 || isempty(beta), beta = 4; end
if isempty(Archive)
    error('selectLeaders:Empty', 'Archive is empty.');
end

% Refresh grid indices
Archive = updateArchive([], Archive, max(numel(Archive), 3));
leaders = cell(3, 1);
usedPos = {};
for i = 1:3
    pool = Archive;
    if ~isempty(usedPos)
        keep = true(numel(pool), 1);
        for j = 1:numel(pool)
            for u = 1:numel(usedPos)
                if isequal(pool(j).Position, usedPos{u})
                    keep(j) = false; break;
                end
            end
        end
        if any(keep)
            pool = pool(keep);
        end
    end
    leaders{i} = archive_utils('select', pool, beta);
    usedPos{end + 1} = leaders{i}.Position; %#ok<AGROW>
end
end

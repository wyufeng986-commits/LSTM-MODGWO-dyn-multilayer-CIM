function varargout = archive_utils(action, varargin)
%ARCHIVE_UTILS  Local grid-archive helpers (ported from MO-IGWO utils).
%   Supported actions:
%     'normalize' | 'update' | 'select' | 'density' | 'roulette'

switch lower(string(action))
    case "normalize"
        varargout{1} = normalize_particles(varargin{1});
    case "update"
        varargout{1} = update_archive(varargin{:});
    case "select"
        varargout{1} = select_leader(varargin{:});
    case "density"
        [a, b] = get_occupied_cells(varargin{1});
        varargout{1} = a; varargout{2} = b;
    case "roulette"
        varargout{1} = roulette(varargin{1});
    otherwise
        error('archive_utils:Action', 'Unknown action %s', action);
end
end

function P = normalize_particles(P)
if isempty(P), return; end
P = P(:);
n = numel(P);
Q = repmat(struct('Position', [], 'SeedSet', [], 'Cost', [], ...
    'F_A', NaN, 'F_B', NaN, 'CostRaw', NaN, 'evalMode', '', ...
    'GridIndex', [], 'GridSubIndex', [], 'detail', struct()), n, 1);
for i = 1:n
    if isfield(P, 'Position') && ~isempty(P(i).Position)
        Q(i).Position = P(i).Position(:)';
    end
    if isfield(P, 'SeedSet') && ~isempty(P(i).SeedSet)
        Q(i).SeedSet = P(i).SeedSet(:)';
    elseif ~isempty(Q(i).Position)
        Q(i).SeedSet = Q(i).Position;
    end
    if isfield(P, 'Cost') && ~isempty(P(i).Cost)
        c = P(i).Cost(:);
        Q(i).Cost = c(1:min(2, numel(c)));
    end
    for f = ["F_A", "F_B", "CostRaw", "evalMode", "GridIndex", "GridSubIndex", "detail"]
        fn = char(f);
        if isfield(P, fn) && ~isempty(P(i).(fn))
            Q(i).(fn) = P(i).(fn);
        end
    end
end
P = Q;
end

function Archive = update_archive(Archive, NewParticles, ArchiveSize)
Archive = normalize_particles(Archive);
NewParticles = normalize_particles(NewParticles);
Total = [Archive; NewParticles];
if isempty(Total)
    Archive = [];
    return;
end

Costs = [Total.Cost];
M = size(Costs, 2);
flag = false(1, M);
for i = 1:M
    for j = 1:M
        if i == j, continue; end
        if dominates(Costs(:, j), Costs(:, i))
            flag(i) = true;
            break;
        end
    end
end
Archive = Total(~flag);
if isempty(Archive), return; end

Costs = [Archive.Cost];
G = create_hypercubes(Costs, 12, 0.1);
for i = 1:numel(Archive)
    [Archive(i).GridIndex, Archive(i).GridSubIndex] = get_grid_index(Archive(i), G);
end
if numel(Archive) > ArchiveSize
    Archive = delete_from_rep(Archive, numel(Archive) - ArchiveSize, 2.5);
end
end

function dom = dominates(x, y)
dom = all(x <= y) && any(x < y);
end

function G = create_hypercubes(costs, ngrid, alpha)
nobj = size(costs, 1);
empty_grid = struct('Lower', [], 'Upper', []);
G = repmat(empty_grid, nobj, 1);
for j = 1:nobj
    min_cj = min(costs(j, :));
    max_cj = max(costs(j, :));
    dcj = alpha * (max_cj - min_cj);
    min_cj = min_cj - dcj;
    max_cj = max_cj + dcj;
    if abs(max_cj - min_cj) < eps
        max_cj = min_cj + 1;
    end
    gx = linspace(min_cj, max_cj, ngrid - 1);
    G(j).Lower = [-inf, gx];
    G(j).Upper = [gx, inf];
end
end

function [Index, SubIndex] = get_grid_index(particle, G)
c = particle.Cost(:);
nobj = numel(c);
ngrid = numel(G(1).Upper);
SubIndex = zeros(1, nobj);
for j = 1:nobj
    U = G(j).Upper;
    i = find(c(j) < U, 1, 'first');
    if isempty(i), i = numel(U); end
    SubIndex(j) = i;
end
if nobj == 1
    Index = SubIndex(1);
else
    Index = sub2ind(ones(1, nobj) * ngrid, SubIndex(1), SubIndex(2));
end
end

function [occ_cell_index, occ_cell_member_count] = get_occupied_cells(pop)
GridIndices = [pop.GridIndex];
occ_cell_index = unique(GridIndices);
occ_cell_member_count = zeros(size(occ_cell_index));
for k = 1:numel(occ_cell_index)
    occ_cell_member_count(k) = sum(GridIndices == occ_cell_index(k));
end
end

function rep = delete_from_rep(rep, EXTRA, gamma)
for k = 1:EXTRA
    [occ_cell_index, occ_cell_member_count] = get_occupied_cells(rep);
    p = occ_cell_member_count .^ gamma;
    p = p / sum(p);
    selected_cell_index = occ_cell_index(roulette(p));
    GridIndices = [rep.GridIndex];
    members = find(GridIndices == selected_cell_index);
    j = members(randi(numel(members)));
    rep(j) = [];
end
end

function leader = select_leader(rep, beta)
if nargin < 2 || isempty(beta), beta = 4; end
if isempty(rep)
    leader = [];
    return;
end
[occ_cell_index, occ_cell_member_count] = get_occupied_cells(rep);
p = occ_cell_member_count .^ (-beta);
p = p / sum(p);
selected_cell_index = occ_cell_index(roulette(p));
GridIndices = [rep.GridIndex];
members = find(GridIndices == selected_cell_index);
leader = rep(members(randi(numel(members))));
end

function i = roulette(p)
p = p(:)' / max(sum(p), eps);
r = rand;
c = cumsum(p);
i = find(r <= c, 1, 'first');
if isempty(i), i = numel(p); end
end

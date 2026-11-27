function downloadData(varargin)
%DOWNLOADDATA  Download CNS layers from Netzschleuder (Figshare mirror).
% Figshare returns 403 in some networks; Netzschleuder CSV zips preserve
% timestamps and are derived from the official CNS release.
%
% Usage:
%   downloadData
%   downloadData('Force', true)

parser = inputParser;
parser.addParameter('Force', false, @(x) islogical(x) || isnumeric(x));
parser.addParameter('ProjectRoot', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opts = parser.Results;

thisDir = fileparts(mfilename('fullpath'));
if strlength(string(opts.ProjectRoot)) == 0
    opts.ProjectRoot = fileparts(thisDir);
end
rawDir = fullfile(char(opts.ProjectRoot), '01_Data', 'raw', 'cns');
if ~exist(rawDir, 'dir')
    mkdir(rawDir);
end

base = 'https://networks.skewed.de/net/copenhagen/files';
files = { ...
    'sms.csv.zip', ...
    'calls.csv.zip', ...
    'bt.csv.zip', ...
    'fb_friends.csv.zip' ...
    };

fprintf('CNS download target: %s\n', rawDir);
for i = 1:numel(files)
    name = files{i};
    outZip = fullfile(rawDir, name);
    layerName = erase(name, '.csv.zip');
    outDir = fullfile(rawDir, layerName);
    if ~opts.Force && exist(outDir, 'dir') && isfile(fullfile(outDir, 'edges.csv'))
        fprintf('[skip] %s already extracted\n', layerName);
        continue;
    end
    fprintf('[get] %s ...\n', name);
    try
        websave(outZip, [base '/' name]);
    catch ME
        warning('downloadData:WebSaveFailed', '%s failed: %s', name, ME.message);
        continue;
    end
    unzip(outZip, outDir);
    fprintf('[ok] %s -> %s\n', name, outDir);
end

metaPath = fullfile(rawDir, 'download_meta.txt');
fid = fopen(metaPath, 'w');
fprintf(fid, 'source=Netzschleuder copenhagen CSV zips\n');
fprintf(fid, 'official=https://doi.org/10.6084/m9.figshare.7267433\n');
fprintf(fid, 'paper=https://doi.org/10.1038/s41597-019-0325-x\n');
fprintf(fid, 'downloadedAt=%s\n', datestr(now, 31));
fprintf(fid, 'note=Figshare may 403; Netzschleuder preserves temporal edges.\n');
fclose(fid);
fprintf('Done. Meta: %s\n', metaPath);
end

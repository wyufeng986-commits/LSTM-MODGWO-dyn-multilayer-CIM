function [occIdx, occCount] = calculateGridDensity(Archive)
%CALCULATEGRIDDENSITY  Occupied objective-space cells and counts.
[occIdx, occCount] = archive_utils('density', Archive);
end

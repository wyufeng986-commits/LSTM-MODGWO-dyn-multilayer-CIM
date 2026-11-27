function Archive = updateArchive(Archive, NewParticles, ArchiveSize)
%UPDATEARCHIVE  Non-dominated Pareto archive with grid truncation.
Archive = archive_utils('update', Archive, NewParticles, ArchiveSize);
end

function visualizePropagation(mcOrResult, outPath)
%VISUALIZEPROPAGATION  Plot dual-side spread curves.
%   Accepts runMonteCarlo output or a single competitiveIC result.

if nargin < 2 || isempty(outPath)
    outPath = '';
end

figure('Visible', 'off');
hold on;
if isfield(mcOrResult, 'meanCurveA')
    x = (1:numel(mcOrResult.meanCurveA))';
    a = mcOrResult.meanCurveA(:);
    b = mcOrResult.meanCurveB(:);
    if isfield(mcOrResult, 'stdCurveA')
        fill([x; flipud(x)], [a - mcOrResult.stdCurveA(:); flipud(a + mcOrResult.stdCurveA(:))], ...
            [0.6 0.8 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
        fill([x; flipud(x)], [b - mcOrResult.stdCurveB(:); flipud(b + mcOrResult.stdCurveB(:))], ...
            [1.0 0.75 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
    end
    plot(x, a, 'b-', 'LineWidth', 1.8, 'DisplayName', 'A (ours)');
    plot(x, b, 'r-', 'LineWidth', 1.8, 'DisplayName', 'B (competitor)');
    title(sprintf('MC competitive IC (R=%d): F_A=%.2f, F_B=%.2f', ...
        mcOrResult.R, mcOrResult.F_A, mcOrResult.F_B));
elseif isfield(mcOrResult, 'history')
    a = mcOrResult.history.sigmaA(:);
    b = mcOrResult.history.sigmaB(:);
    x = (1:numel(a))';
    plot(x, a, 'b-', 'LineWidth', 1.8, 'DisplayName', 'A (ours)');
    plot(x, b, 'r-', 'LineWidth', 1.8, 'DisplayName', 'B (competitor)');
    title(sprintf('Single-run IC: \\sigma_A=%d, \\sigma_B=%d', ...
        mcOrResult.sigmaA, mcOrResult.sigmaB));
else
    error('visualizePropagation:Input', 'Unrecognized result struct.');
end
xlabel('Propagation step');
ylabel('Activated nodes');
legend('Location', 'best');
grid on;
hold off;

if strlength(string(outPath)) > 0
    outDir = fileparts(outPath);
    if strlength(string(outDir)) > 0 && ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(gcf, outPath);
end
end

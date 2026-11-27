function X = extractTemporalFeatures(sigmaA, sigmaB, N)
%EXTRACTTEMPORALFEATURES  Per-step dynamic features for LSTM.
%   X(:,t) = [sigA, sigB, dA, dB, unactivated, cA, cB]

sigmaA = sigmaA(:)';
sigmaB = sigmaB(:)';
T = numel(sigmaA);
dA = [sigmaA(1), diff(sigmaA)];
dB = [sigmaB(1), diff(sigmaB)];
U = max(N - sigmaA - sigmaB, 0);
cA = sigmaA / max(N, 1);
cB = sigmaB / max(N, 1);
X = [sigmaA; sigmaB; dA; dB; U; cA; cB]; % 7 x T
end

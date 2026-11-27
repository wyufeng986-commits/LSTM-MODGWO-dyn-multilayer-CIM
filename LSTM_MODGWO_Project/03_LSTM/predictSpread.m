function [Yhat, YhatSeq] = predictSpread(model, Xin)
%PREDICTSPREAD  Predict remaining dual-side spread sequence.
%   Xin: F x r raw OR already-normalized sequence
%   YhatSeq: 2 x horizon (denormalized)
%   Yhat: struct with sigmaA, sigmaB, finalA, finalB

X = (Xin - model.muX) ./ model.sigX;
y = predict(model.net, {X});
if iscell(y), y = y{1}; end
y = y(:) .* model.sigY + model.muY;

H = model.horizon;
Ymat = reshape(y, 2, H);
% physical constraints
Ymat = max(Ymat, 0);
% enforce nondecreasing soft fix by cummax optional - keep mild clip only
Nproxy = max([Ymat(1, :), Ymat(2, :), 1]);
for t = 1:H
    s = Ymat(1, t) + Ymat(2, t);
    if s > Nproxy * 2 % loose; better use true N if available
        % no-op unless absurd
    end
end

YhatSeq = Ymat;
Yhat = struct();
Yhat.sigmaA = Ymat(1, :);
Yhat.sigmaB = Ymat(2, :);
Yhat.finalA = Ymat(1, end);
Yhat.finalB = Ymat(2, end);
end

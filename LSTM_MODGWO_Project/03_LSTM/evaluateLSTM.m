function metrics = evaluateLSTM(model, data, splitName)
%EVALUATELSTM  MAE/RMSE/R2 on a prepared split (denormalized space).

if nargin < 3, splitName = 'test'; end
switch lower(splitName)
    case 'train'
        Xc = data.XTrain; Ym = data.YTrain;
    case 'val'
        Xc = data.XVal; Ym = data.YVal;
    otherwise
        Xc = data.XTest; Ym = data.YTest;
end

n = numel(Xc);
if n == 0
    metrics = struct('n', 0); return;
end

H = model.horizon;
Ytrue = zeros(2, H, n);
Ypred = zeros(2, H, n);
for i = 1:n
    yhat = predict(model.net, Xc{i});
    yhat = yhat(:) .* model.sigY + model.muY;
    ytrue = Ym(i, :)' .* data.sigY + data.muY;
    Ypred(:, :, i) = reshape(yhat, 2, H);
    Ytrue(:, :, i) = reshape(ytrue, 2, H);
end

err = Ypred - Ytrue;
mae = mean(abs(err), 'all');
rmse = sqrt(mean(err.^2, 'all'));
ssRes = sum(err(:).^2);
ssTot = sum((Ytrue(:) - mean(Ytrue(:))).^2);
r2 = 1 - ssRes / max(ssTot, eps);

metrics = struct();
metrics.split = splitName;
metrics.n = n;
metrics.MAE = mae;
metrics.RMSE = rmse;
metrics.R2 = r2;
metrics.finalA_MAE = mean(abs(Ypred(1, end, :) - Ytrue(1, end, :)), 'all');
metrics.finalB_MAE = mean(abs(Ypred(2, end, :) - Ytrue(2, end, :)), 'all');
metrics.finalA_RMSE = sqrt(mean((Ypred(1, end, :) - Ytrue(1, end, :)).^2, 'all'));
metrics.finalB_RMSE = sqrt(mean((Ypred(2, end, :) - Ytrue(2, end, :)).^2, 'all'));
metrics.Yt = Ytrue;
metrics.Yp = Ypred;
end

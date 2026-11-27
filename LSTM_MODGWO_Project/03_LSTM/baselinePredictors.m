function metrics = baselinePredictors(data, name)
%BASELINEPREDICTORS  Simple baselines on prepared splits (test metrics).
% name: 'persistence' | 'linear' | 'svr'

name = lower(string(name));
H = data.horizon;
nTrain = size(data.YTrain, 1);
nTest = size(data.YTest, 1);

denYrow = @(row) row(:) .* data.sigY + data.muY;
denX = @(X) X .* data.sigX + data.muX;

switch name
    case "persistence"
        Yp = zeros(2, H, nTest);
        Yt = zeros(2, H, nTest);
        for i = 1:nTest
            X = denX(data.XTest{i});
            a0 = X(1, end); b0 = X(2, end);
            Yp(1, :, i) = a0;
            Yp(2, :, i) = b0;
            Yt(:, :, i) = reshape(denYrow(data.YTest(i, :)), 2, H);
        end
    case "linear"
        F = design_matrix(data.XTrain);
        Ymat = zeros(nTrain, data.responseDim);
        for i = 1:nTrain
            Ymat(i, :) = denYrow(data.YTrain(i, :))';
        end
        beta = F \ Ymat;
        Ft = design_matrix(data.XTest);
        Yhat = Ft * beta;
        Yp = zeros(2, H, nTest);
        Yt = zeros(2, H, nTest);
        for i = 1:nTest
            Yp(:, :, i) = reshape(Yhat(i, :)', 2, H);
            Yt(:, :, i) = reshape(denYrow(data.YTest(i, :)), 2, H);
        end
    case "svr"
        F = design_matrix(data.XTrain);
        yA = zeros(nTrain, 1); yB = zeros(nTrain, 1);
        for i = 1:nTrain
            Yi = reshape(denYrow(data.YTrain(i, :)), 2, H);
            yA(i) = Yi(1, end); yB(i) = Yi(2, end);
        end
        mdlA = fitrsvm(F, yA, 'KernelFunction', 'gaussian', 'Standardize', true);
        mdlB = fitrsvm(F, yB, 'KernelFunction', 'gaussian', 'Standardize', true);
        Ft = design_matrix(data.XTest);
        pA = predict(mdlA, Ft);
        pB = predict(mdlB, Ft);
        Yp = zeros(2, H, nTest);
        Yt = zeros(2, H, nTest);
        for i = 1:nTest
            Yp(1, :, i) = pA(i);
            Yp(2, :, i) = pB(i);
            Yt(:, :, i) = reshape(denYrow(data.YTest(i, :)), 2, H);
        end
    otherwise
        error('baselinePredictors:Name', 'Unknown baseline %s', name);
end

err = Yp - Yt;
mae = mean(abs(err), 'all');
rmse = sqrt(mean(err.^2, 'all'));
r2 = 1 - sum(err(:).^2) / max(sum((Yt(:) - mean(Yt(:))).^2), eps);

metrics = struct('name', char(name), 'n', nTest, ...
    'MAE', mae, 'RMSE', rmse, 'R2', r2, ...
    'finalA_MAE', mean(abs(Yp(1,end,:) - Yt(1,end,:)), 'all'), ...
    'finalB_MAE', mean(abs(Yp(2,end,:) - Yt(2,end,:)), 'all'));
end

function F = design_matrix(Xcells)
n = numel(Xcells);
d = size(Xcells{1}, 1);
F = zeros(n, d + 1);
for i = 1:n
    F(i, :) = [1, Xcells{i}(:, end)'];
end
end

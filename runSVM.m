N = 20;
data = zeros(2*N, 4);
data(1:N, 1) = randn(N,1) * 3 + 2;
data(1:N, 2) = randn(N,1) * 2;
data(1:N, 3) = randn(N,1) * 1.5 + 1;
data(1:N, 4) = ones(N,1);
data(N+1:end, 1) = randn(N,1) * 2 + 4;
data(N+1:end, 2) = randn(N,1) + 2;
data(N+1:end, 3) = randn(N,1) - 1;
data(N+1:end, 4) = ones(N,1) * (-1);

feature_dim = 3;
feature_num = 2*N;
class1_num = N;
class2_num = N;
features = data(:, 1:3);
classifications = data(:, 4);
c = 0.1;
[z, e] = SVM(features, classifications, c, feature_num, feature_dim);

scatter3(data(1:class1_num, 1), data(1:class1_num, 2), data(1:class1_num, 3), 'ob');
hold on
scatter3(data(class1_num + 1: end, 1), data(class1_num + 1: end, 2), data(class1_num + 1: end, 3), 'og');
[X, Y] = meshgrid(floor(min(data(:,1))): ceil(max(data(:,1))), floor(min(data(:,2))): ceil(max(data(:,2))));
Z = (-z(1)/z(3)) * X - (z(2)/z(3)) * Y - z(4)/z(3);
surf(X, Y, Z);
hold off

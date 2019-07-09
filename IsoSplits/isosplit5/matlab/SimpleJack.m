d = 20;                                              % number of dimensions
n = 1e4;                                             % points per cluster
K = 10;                                              % number of clusters
rng(1);                                              % fix the seed
X = randn(d,K*n) + 2.0*kron(randn(d,K),ones(1,n));   % N=K*n points in d dims
L = isosplit5_mex(X);                                % cluster: takes 2 sec
k = mode(reshape(L,[n,K]),1);                        % get gross labeling
fprintf('gross label errors: %d\n',sum(sort(k)-(1:K)))
fprintf('number of points misclassified: %d\n',sum(L~=kron(k,ones(1,n))))


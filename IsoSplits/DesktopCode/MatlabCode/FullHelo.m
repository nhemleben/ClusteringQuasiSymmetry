function FullHelo(A,Y, N, HasLegend)

path0=fileparts(mfilename('fullpath'));   % directory of this script
addpath([path0,'/matlab']);               % provides isosplit5_mex
addpath([path0,'/matlab/visualization']); % provides view_clusters()

B= cell2mat(A);
X= cell2mat(Y);
%Need to fix dimensions:

n= length(B)/N;
A = zeros(n,N );
for i= 0: n-1
    A(i+1,:) = B(i*N+1: (i+1)*N);
end
n= length(X)/N;
Y = zeros(n,N );
for i= 0: n-1
    Y(i+1,:) = X(i*N+1: (i+1)*N);
end

%Note that the actual space Y is the one used for labeling
%and A the plot variables is used for ploting
% Run the clustering
labels=isosplit5_mex(Y);

% Display the results
figure;
if HasLegend
    view_clusters(A,labels);
else
    view_clusters_no_legend(A,labels);
end


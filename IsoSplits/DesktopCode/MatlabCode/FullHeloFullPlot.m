function FullHelo(A,Y,T, N, HasLegend)

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

T= string(T)
i=1
view_clusters_no_legend(A(i:i+1,:),labels);
title( strcat( T(i+1) , " vs ", T(i) )  )
saveas(gcf,'UnitIsoPlot.png')

% Display the results
%T= string(T)
%for i= 1:n-1
%    figure;
%    if HasLegend
%        view_clusters(A(i:i+1,:),labels);
%        title(T(i+1) , ' VS ', T(i) )
%    else
%        view_clusters_no_legend(A(i:i+1,:),labels);
%        title( strcat( T(i+1) , " vs ", T(i) )  )
%    end
%end




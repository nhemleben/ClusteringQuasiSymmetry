function FullHelo(A,Y,T, N, HasLegend)

path0=fileparts(mfilename('fullpath'));   % directory of this script
addpath([path0,'/matlab']);               % provides isosplit5_mex
addpath([path0,'/matlab/visualization']); % provides view_clusters()

B= cell2mat(A);
X= cell2mat(Y);
%Need to fix dimensions:

n= length(X)/N;
Y = zeros(n,N );
for i= 0: n-1
    Y(i+1,:) = X(i*N+1: (i+1)*N);
end
n= length(B)/N;
A = zeros(n,N );
for i= 0: n-1
    A(i+1,:) = B(i*N+1: (i+1)*N);
end

%Note that the actual space Y is the one used for labeling
%and A the plot variables is used for ploting
% Run the clustering
labels=isosplit5_mex(Y);

% Display the results
T= string(T)
for i= 1:n-2
    fig=figure;
    if HasLegend
        title( strcat(T(i) , " vs ", T(i+1)  , " vs ", T(i+2) ) )
        view_clusters(  A(i:i+2,:) ,labels);
    else
        Titl=  strcat( T(i) , " vs ", T(i+1) ," vs ", T(i+2) )
        view_clusters_no_legend( A(i:i+2,:) ,labels);
        title(Titl )


 %       saveas(fig, strcat(Titl , ".fig") )
    end
end


%This method didn't work for some reason (probably too many dimensions in plot
%also doesn't really matter
%for i= 1:n-1:2
%    figure;
%    if HasLegend
%        view_clusters([ A(1,:) ; A(i:i+2,:)] ,labels);
%        title( strcat("Iota vs", T(i) , " vs ", T(i+1) )  )
%    else
%        view_clusters_no_legend([A(1,:) ; A(i:i+1,:)] ,labels);
%        title( strcat("Iota vs ", T(i) , " vs ", T(i+1) )  )
%    end
%end
%
%view_clusters([ A(1,:) ; A(n-1:n,:)] ,labels);
%title( strcat("Iota vs ", T(n-1) , " vs ", T(n) )  )

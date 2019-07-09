function PlotClusters(A,T, N, HasLegend)

%path0=fileparts(mfilename('fullpath'));   % directory of this script
%addpath([path0,'/matlab']);               % provides isosplit5_mex
%addpath([path0,'/matlab/visualization']); % provides view_clusters()


disp('Inside Matlab')
B= cell2mat(A);
disp('Finised cell')


n= length(B)/N;
A = zeros(n,N );
for i= 0: n-1
    A(i+1,:) = B(i*N+1: (i+1)*N);
end

%Note that the actual space Y is the one used for labeling
%and A the plot variables is used for ploting

T= string(T);

save('saveData.mat','A');
save('saveVar.mat','T');

example = matfile('saveLabels.mat');
labels = example.labels;

count= zeros(max(labels));

for i=1: max(labels)
    mask = labels == i;
    count(i) = sum(mask);
    A1 = A(:,mask);
    [coeff,score,latent] = pca(A1');
    disp(strcat("Cluster Number " , num2str(i), ", Number of points " , num2str(sum(mask)) ) )
    disp( coeff )
    disp( latent )
    disp(' ')
end

figure
plot(count)
figure
histogram(count)

%Show figure
%figure
scatter(A(1,:),A(1+1,:),'.')
%title( strcat( T(1+1) , " vs ", T(1) )  )


%for i =1:n-1
%    figure
%    scatter(A(i,:),A(i+1,:),'.')
%    title( strcat( T(i+1) , " vs ", T(i) )  )
%end




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




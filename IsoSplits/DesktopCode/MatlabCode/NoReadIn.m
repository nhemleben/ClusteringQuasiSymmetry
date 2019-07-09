function NoReadIn()

%disp('Inside Matlab')

%Note that the actual space Y is the one used for labeling
%and A the plot variables is used for ploting
%save('saveData.mat','A');
%save('saveVar.mat','T');

example = matfile('saveData.mat');
A = example.A;

example = matfile('saveVar.mat');
T = example.T;

example = matfile('saveLabels.mat');
labels = example.labels;

%disp('Finised Mat Files')


count= zeros(max(labels),1);

for i=1: max(labels)
    mask = labels == i;
    count(i) = sum(mask);
%    A1 = A(:,mask);
%    [coeff,score,latent] = pca(A1');
%    disp(strcat("Cluster Number " , num2str(i), ", Number of points " , num2str(sum(mask)) ) )
%    disp( coeff )
%    disp( latent )
%    disp(' ')
end

Count = sort(count);
%disp(Count)
%figure
%plot(Count)
%figure
%histogram(Count)

PercClust= Count/ (sum(Count));
%disp(PercClust)
%figure
%histogram(PercClust)

CumPerc = cumsum( flipud( PercClust) );

%figure
%plot(CumPerc)



%figure
%plot(count)
%figure
%histogram(count)

%get 15 largest clusters and their indexes
[B, I] = maxk(count, 15);

totalmask= 0*labels;

for i=1: length(I)
    mask = labels == I(i);
    A1 = A(:,mask);
    totalmask= totalmask+mask;

%block of ploting code
%    figure
%    scatter(A1(2,:),A1(11,:),'.')
%    title( strcat( T(11) , " vs ", T(2) )  )

    [coeff,score,latent, tsquared, explained, mu] = pca(A1');
    disp(strcat("Cluster Number " , num2str(i), ", Number of points " , num2str(sum(mask)) ) )
    disp( coeff( :, 1:5))
    disp('Percent of Variance Explained')
    disp( explained(1:5))
    disp('Average Coordinates of Cluster')
    disp( mu )
    disp(' ')
end


A1 = A(:,logical(totalmask));
figure
hold on
scatter(A(2,:),A(11,:),'.', 'b')
scatter(A1(2,:),A1(11,:),'.','r')
title( strcat( T(11) , " vs ", T(2)  , " Top Clusters (red) Vs All (blue)")  )



%figure
%scatter(A(2,:),A(11,:),'.')
%title( strcat( T(11) , " vs ", T(2)  , " All points" ) )



[coeff,score,latent, tsquared, explained, mu] = pca(A');
disp(strcat("Cluster Number ALL" , " Number of points " , num2str(length(A)) ) )
disp( coeff( :, 1:5))
disp('Percent of Variance Explained')
disp( explained(1:5))
disp('Average Coordinates of Cluster')
disp( mu )
disp(' ')



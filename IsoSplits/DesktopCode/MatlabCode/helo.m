function helo(X1,X2,X3)
%function helo(A)
%B1 = cell2mat(A[1])
%B2 = cell2mat(A[2])
%B3 = cell2mat(A3)


path0=fileparts(mfilename('fullpath'));   % directory of this script
addpath([path0,'/matlab']);               % provides isosplit5_mex
addpath([path0,'/matlab/visualization']); % provides view_clusters()

% Generate the data
%X=A
X1= cell2mat(X1);
X2= cell2mat(X2);
X3= cell2mat(X3);

%X=cat(1,X1,X2);
X=cat(1,X1,X2,X3);
%disp(X)
%labels=cat(2,ones(1,size(X1,2)*1),ones(1,size(X2,2))*2);
%labels=cat(2,ones(1,size(X1,2)*1),ones(1,size(X2,2))*2,ones(1,size(X3,2))*3);

% Run the clustering
labels=isosplit5_mex(X);


% Display the results
figure;
view_clusters(X,labels);


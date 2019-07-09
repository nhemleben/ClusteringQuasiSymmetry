function UnitIsoCluster()

path0=fileparts(mfilename('fullpath'));   % directory of this script
addpath([path0,'/matlab']);               % provides isosplit5_mex
addpath([path0,'/matlab/visualization']); % provides view_clusters()

load('Variables.mat')
load('Keys.mat')
%List of all variables in 'alphabetical' order (capitals count first)
B = [
%R0c1          ; %Just using outputs
%R0c2          ;
%R0c3          ;
%Z0s1          ;
%Z0s2          ;
%Z0s3          ;
dominantnfps  ;
etabar        ;
helicities    ; %had issues with a row of all zeros when inverting something
iotas         ;
maxcurvatures ;
maxelongations;
maxmodBinv    ;
nfps          ;
rmscurvatures ;
stdofR        ;
stdofZ  ];

C = unique(B', 'rows');
A= C';


%Note that the actual space Y is the one used for labeling
%and A the plot variables is used for ploting
% Run the clustering
labels=isosplit5_mex(A);
[n,m]= size(A);
T= string( Names )
i=10
view_clusters_no_legend(A(4:5,:),labels);
title( strcat( T(i+1,:) , ' vs ', T(i,:) )  )
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




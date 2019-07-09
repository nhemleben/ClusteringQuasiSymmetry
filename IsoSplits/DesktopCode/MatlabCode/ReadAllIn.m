
function ReadAllIn(A,T, N)

path0=fileparts(mfilename('fullpath'));   % directory of this script
addpath([path0,'/matlab']);               % provides isosplit5_mex
addpath([path0,'/matlab/visualization']); % provides view_clusters()



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


save('saveDataFull.mat','A');
save('saveVarFull.mat','T');

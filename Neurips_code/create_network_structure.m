clear
close all
clc

addpath('../../matlab_simulations');

settingName = 'Circular';
% settingName =  'Connected';
% settingName = 'Barbell';
% settingName = 'Disconnected';
numProcesses = 10;

[W, Lap] = getNetworkWeights(settingName, numProcesses);
Wreg = 3.1;
% Wreg = 2.898;
if(strcmp(settingName, 'Connected'))
    Wreg = 0.000000001;
end
% Wreg = 10;
W = (Wreg*eye(size(W)) + W)/(Wreg+1); %Regularize

Weig = sort(eig(W),'ascend');
WN = Weig(1);
if(WN <=2/3)
    error('Error in W');
end


return
%%


fid = fopen(['../network_settings/' settingName '_' num2str(numProcesses) '.txt'],'w');
for i = 1:size(W,1)
    for j = 1:size(W,2)
        fprintf(fid,'%2.10f\n',W(i,j));
    end
end

fid = fopen(['../network_settings/L' settingName '_' num2str(numProcesses) '.txt'],'w');
for i = 1:size(Lap,1)
    for j = 1:size(Lap,2)
        fprintf(fid,'%2.10f\n',Lap(i,j));
    end
end


fclose(fid);








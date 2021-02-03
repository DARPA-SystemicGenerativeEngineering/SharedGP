%Demostration of share-gplvm across multiple data space.
% NOTE: initilization of the latent could matter. 
% Now provide PCA/Isomap/share-isomap initilization. default is share-isomap. 
% Changes can be made in function:train_scigplvm_v2
% 
% Author: Wei Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 02-Feb-2021

clear
addpath(genpath('./library'))
addpath(genpath('./codes'))
addpath(genpath('./util'))
%% prepare data
load('Design_data_mini.mat')

IMG_SIZE1 = 60;
IMG_SIZE2 = 60;

y1 = vmis_stress';
y2 = xs';
y3 = compliance';
y4 = design_parameters';

nTr = 200;
nTe = 100;
idAll = randperm(size(y4,1));

idTr = idAll(1:nTr);
idTe = idAll(size(y4,1):-1:size(y4,1)+1-nTe);

Y{1} = y1(idTr,:);
Y{2} = y2(idTr,:);
Y{3} = y3(idTr,:);
Y{4} = y4(idTr,:);

Yte{1} = y1(idTe,:);
Yte{2} = y2(idTe,:);
Yte{3} = y3(idTe,:);
Yte{4} = y4(idTe,:);


%% train model with training data
rank = 2; %latent dimension
% model = train_scigplvm_v2(Y,rank,'ard');
% model = train_scigplvm_v2(Y,rank,'ard');
model = train_scigplvm_dpp_v2(Y,rank,'ard');

%% predicting all y and U (latent) given y1 (fast approach)
model2 = sgplvm_invGp_v1(model,1,Yte{1});

model2.u_star   %predicted latent variables
model2.y_star   %predicted y

figure(1)
plot(Yte{3},model2.y_star{3},'r+')
axis equal
title('Truths vs. Predictions')

%% predicting all y and U (latent) given y2. (improved approach)

% quick inference without dpp
% model3 = scigplvm_infere_v3(model,2,y2(idTe,:));  

% full inference
model3 = train_scigplvm_dpp_infere_v4(model,2,y2(idTe,:));

model3.u_star;  %predicted latent variables
model3.y_star;  %predicted y

figure(2)
plot(Yte{3},model3.y_star{3},'r+')
axis equal
title('Truths vs. Predictions')
%% predicting all y given U (latent)
model4 = sgplvm_pred(model,model3.u_star);
model4.yNew %predicted y

figure(3)
plot(Yte{3},model4.yNew{3},'r+')
axis equal
title('Truths vs. Predictions')

%% viz
rank = 2;
% model_dpp = train_scigplvm_dpp_v2(Y,rank,'ard');
model_dpp = model;

marker = {'r*','bo','k^','gd'};
% get latent label and give it colormap
[val,loc] = max(model_dpp.stat.dp_phi{1, 1}');
marker = parula(max(loc));

figure(4)
clf
hold on 
for i = 1:max(loc)
%     scatter(model2.U((loc==i),1),model2.U((loc==i),2),marker{i});
    scatter(model_dpp.U((loc==i),1),model_dpp.U((loc==i),2),30,marker(i,:),'filled');
%     scatter3(model2.U((loc==i),1),model2.U((loc==i),2),model2.U((loc==i),3), 10,marker(i,:));
end



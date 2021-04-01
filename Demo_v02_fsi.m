%Demostration of share-gplvm across multiple data space.
% NOTE: initilization of the latent could matter. 
% Now provide PCA/Isomap/share-isomap initilization. default is share-isomap. 
% Changes can be made in function:train_scigplvm_v2
% 
% Author: Wei Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 02-Feb-2021
close all
clc
clear
% set(gca, 'FontName', 'Arial', 'FontSize', 18)
%% data directory
data_dir = './fsi_data/results_3/'
set(0, 'DefaultAxesFontSize', 18);
addpath(genpath('./library'))
addpath(genpath('./codes'))
addpath(genpath('./util'))
addpath(genpath(data_dir))
%% prepare data
load('Design_data_mini.mat')
load('400_flow_data.mat')
load('400_solid_data.mat')
load('400_parameters.mat')
nTr = 25;
nTe = 50;
%% check filed
% figure(100)
% pointsize=10;
% nsample = 1;
% scatter(flow_coord_x(nsample,:), flow_coord_y(nsample,:), pointsize, velocity_x(nsample,:));
% cb = colorbar();

IMG_SIZE1 = 60;
IMG_SIZE2 = 60;

% y1 = vmis_stress';
% y2 = xs';
% y3 = compliance';
% y4 = design_parameters';
% 
rank = 4; %latent dimension


% y1 = velocity_x;
% y2 = velocity_y;
% y3 = pressure;

% y1 = disp_x;
% y2 = disp_y;
% y3 = disp_y;
% y3 = von_mises_stress;
% y3 = Sxx;
% y1 = Syy;
% y2 = Sxy;

% y3 = Sxx;
% y1 = Syy;
% y2 = Sxy;
% y5 = von_mises_stress;

y1 = velocity_x;
y2 = velocity_y;
y3 = pressure;
y5 = disp_x;
y6 = disp_y;
y7 = von_mises_stress;

% y1_vec = reshape(y1,[],1);
% y2_vec = reshape(y2,[],1);
% y3_vec = reshape(y3,[],1);
% y5_vec = reshape(y5,[],1);
% y6_vec = reshape(y6,[],1);

% y1 = (y1-mean(y1_vec))/std(y1_vec);
% y2 = (y2-mean(y2_vec))/std(y2_vec);
% y3 = (y3-mean(y3_vec))/std(y3_vec);

% y1 = (y1-mean(y1_vec))/std(y1_vec);
% y2 = (y2-mean(y2_vec))/std(y2_vec);
% y3 = (y3-mean(y3_vec))/std(y3_vec);
% y5 = (y5-mean(y5_vec))/std(y5_vec);
% y6 = (y6-mean(y6_vec))/std(y6_vec);

% y1 = (y1-mean(y1))./std(y1);
% y2 = (y2-mean(y2))./std(y2);
% y3 = (y3-mean(y3))./std(y3);
% y5 = (y5-mean(y5))/std(y5);

y1 = y1/max(max(y1));
y2 = y2/max(max(y2));
y3 = y3/max(max(y3));

y5 = y5/max(max(y5));
y6 = y6/max(max(y6));
y7 = y7/max(max(y7));

inlet_vel_x = inlet_vel_x ./ max(inlet_vel_x);
inlet_vel_y = inlet_vel_y ./ max(inlet_vel_y);
elastic_modulus = elastic_modulus./max(elastic_modulus);
poisson_ratio = poisson_ratio./max(poisson_ratio);

% y4 = [(inlet_vel_x'-mean(inlet_vel_x'))/std(inlet_vel_x),...
%     (inlet_vel_y'- mean(inlet_vel_y'))/std(inlet_vel_y)];
y4 = [(inlet_vel_x'-mean(inlet_vel_x'))/std(inlet_vel_x),...
    (inlet_vel_y'- mean(inlet_vel_y'))/std(inlet_vel_y),...
    (elastic_modulus'-mean(elastic_modulus))/std(elastic_modulus),...
    (poisson_ratio'- mean(poisson_ratio))/std(poisson_ratio)];
% y4 = [inlet_vel_x'./max(inlet_vel_x'), inlet_vel_y'./max(inlet_vel_y'),...
%     elastic_modulus'./max(elastic_modulus),...
%     poisson_ratio'./max(elastic_modulus)] ;
% nTr = 200;
% nTe = 100;
idAll = randperm(size(y4,1));

idTr = idAll(1:nTr);
idTe = idAll(size(y4,1):-1:size(y4,1)+1-nTe);

% Y{1} = y1(idTr,:);
% Y{2} = y2(idTr,:);
% Y{3} = y3(idTr,:);
% Y{4} = y4(idTr,:);
% 
% Yte{1} = y1(idTe,:);
% Yte{2} = y2(idTe,:);
% Yte{3} = y3(idTe,:);
% Yte{4} = y4(idTe,:);

Y{1} = y1(idTr,:);
Y{2} = y2(idTr,:);
Y{3} = y3(idTr,:);
Y{4} = y4(idTr,:);
Y{5} = y5(idTr,:);
Y{6} = y6(idTr,:);
Y{7} = y7(idTr,:);


Yte{1} = y1(idTe,:);
Yte{2} = y2(idTe,:);
Yte{3} = y3(idTe,:);
Yte{4} = y4(idTe,:);
Yte{5} = y5(idTe,:);
Yte{6} = y6(idTe,:);
Yte{7} = y7(idTe,:);

%% train model with training data
% rank = 4; %latent dimension
% model = train_scigplvm_v2(Y,rank,'ard');
% model = train_scigplvm_v2(Y,rank,'ard');
model = train_scigplvm_dpp_v2(Y,rank,'ard');

% figure(1)
% plot(model.train_pred{4}, model.yTr{4},'r+')
% % plot(model.train_pred{4}(:,1), model.yTr{4}(:,1),'r+')
% xlim([-1 3])
% ylim([-1 3])
% axis equal
% title('Truths vs. Predictions')
% % saveas(gcf,strcat(data_dir,'/figure1.png'))

%% predicting all y and U (latent) given y1 (fast approach)--inference
% model2 = sgplvm_invGp_v1(model,1,Yte{1});
% % model2 = sgplvm_invGp_v1(model,4,Yte{4});
% 
% model2.u_star;   %predicted latent variables
% model2.y_star;   %predicted y
% 
% figure(3)
% plot(Yte{3},model2.y_star{3},'r+')
% axis equal
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure3.png')))
% 
% figure(1)
% plot(Yte{1},model2.y_star{1},'r+')
% axis equal
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure1.png')))
% 
% 
% figure(2)
% plot(Yte{2},model2.y_star{2},'r+')
% axis equal
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure2.png')))
% 
% figure(5)
% plot(Yte{5},model2.y_star{5},'r+')
% axis equal
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure5.png')))
% 
% figure(6)
% plot(Yte{6},model2.y_star{6},'r+')
% axis equal
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure6.png')))
% 
% % mdl = fitlm(Yte{2}(90,:),model2.y_star{2}(90,:))
% % figure(3)
% % plot(Yte{4},model2.y_star{4},'r+')
% % axis equal
% % title('Truths vs. Predictions')
% 
% figure(4)
% plot(Yte{4}(:,1),model2.y_star{4}(:,1),'r+', Yte{4}(:,2),model2.y_star{4}(:,2),'ko',...
%     Yte{4}(:,3),model2.y_star{4}(:,3),'g+', Yte{4}(:,4),model2.y_star{4}(:,4),'m*')
% % plot(Yte{4}(:,1),model2.y_star{4}(:,1),'r+', Yte{4}(:,2),model2.y_star{4}(:,2),'ko')
% axis equal
% legend('inlet\_vel\_x','inlet\_vel\_y', 'elastic', 'poisson','location','northwest')
% title('Truths vs. Predictions')
% saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure4.png')))

%% predicting all y and U (latent) given y2. (improved approach)

% quick inference without dpp
% model3 = scigplvm_infere_v3(model,2,y2(idTe,:));  

% full inference
model3 = train_scigplvm_dpp_infere_v4(model,1,y1(idTe,:));
% model3 = train_scigplvm_dpp_infere_v4(model,4,y4(idTe,:));

model3.u_star;  %predicted latent variables
model3.y_star;  %predicted y

r_square=[1];
for out=2:7
    mdl = fitlm(reshape(Yte{out}',1,[]),reshape(model3.y_star{out}',1,[]));
    r_square(out) = mdl.Rsquared.Ordinary;
end

figure(12)
plot(Yte{2},model3.y_star{2},'r+')
axis equal
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure12.png')))

figure(13)
plot(Yte{3},model3.y_star{3},'r+')
axis equal
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure13.png')))


figure(14)
plot(Yte{4}(:,1),model3.y_star{4}(:,1),'r+', Yte{4}(:,2),model3.y_star{4}(:,2),'ko',...
    Yte{4}(:,3),model3.y_star{4}(:,3),'g+', Yte{4}(:,4),model3.y_star{4}(:,4),'m*')
% plot(Yte{4}(:,1),model2.y_star{4}(:,1),'r+', Yte{4}(:,2),model2.y_star{4}(:,2),'ko')
axis equal
legend('inlet\_vel\_x','inlet\_vel\_y', 'elastic', 'poisson','location','northwest')
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure14.png')))


figure(15)
plot(Yte{5},model3.y_star{5},'r+')
axis equal
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure15.png')))

figure(16)
plot(Yte{6},model3.y_star{6},'r+')
axis equal
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure16.png')))

figure(17)
plot(Yte{7},model3.y_star{7},'r+')
axis equal
title('Truths vs. Predictions')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure17.png')))

set(0, 'DefaultAxesFontSize', 24);
pointsize = 15;
figure(22)
nsample = 1;
subplot(3,1,1)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, Yte{2}((nsample),:));
cb=colorbar();
title('Truth')
subplot(3,1,2)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, model3.y_star{2}((nsample),:));
cb=colorbar();
title('GPLVM prediction')
subplot(3,1,3)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, Yte{2}((nsample),:) - model3.y_star{2}((nsample),:));
cb=colorbar();
title ('Difference')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure22.png')))


figure(23)
nsample = 1;
subplot(3,1,1)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, Yte{3}((nsample),:));
cb=colorbar();
title('Truth')
subplot(3,1,2)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, model3.y_star{3}((nsample),:));
cb=colorbar();
title('GPLVM prediction')
subplot(3,1,3)
scatter(flow_coord_x((nsample),:), flow_coord_y((nsample),:), pointsize, Yte{3}((nsample),:) - model3.y_star{3}((nsample),:));
cb=colorbar();
title ('Difference')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure23.png')))


figure(25)
nsample = 1;
subplot(1,3,1)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{5}((nsample),:));
cb=colorbar();
title('Truth')
subplot(1,3,2)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, model3.y_star{5}((nsample),:));
cb=colorbar();
title('GPLVM prediction')
subplot(1,3,3)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{5}((nsample),:) - model3.y_star{5}((nsample),:));
cb=colorbar();
title ('Difference')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure25.png')))


figure(26)
nsample = 1;
subplot(1,3,1)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{6}((nsample),:));
cb=colorbar();
title('Truth')
subplot(1,3,2)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, model3.y_star{6}((nsample),:));
cb=colorbar();
title('GPLVM prediction')
subplot(1,3,3)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{6}((nsample),:) - model3.y_star{6}((nsample),:));
cb=colorbar();
title ('Difference')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure26.png')))

figure(27)
nsample = 1;
subplot(1,3,1)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{7}((nsample),:));
cb=colorbar();
title('Truth')
subplot(1,3,2)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, model3.y_star{7}((nsample),:));
cb=colorbar();
title('GPLVM prediction')
subplot(1,3,3)
scatter(solid_coord_x((nsample),:), solid_coord_y((nsample),:), pointsize, Yte{7}((nsample),:) - model3.y_star{7}((nsample),:));
cb=colorbar();
title ('Difference')
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_figure27.png')))


model.ker_params{2}.l

%% predicting all y given U (latent)
model4 = sgplvm_pred(model,model3.u_star);
model4.yNew %predicted y

figure(3)
plot(Yte{3},model4.yNew{3},'r+')
axis equal
title('Truths vs. Predictions')

%% viz
% rank = 2;
% model_dpp = train_scigplvm_dpp_v2(Y,rank,'ard');
model_dpp = model;

marker = {'r*','bo','k^','gd'};
% get latent label and give it colormap
[val,loc] = max(model_dpp.stat.dp_phi{1, 1}');
marker = parula(max(loc));

figure(10)
clf
% axis equal
hold on 
for i = 1:max(loc)
%     scatter(model2.U((loc==i),1),model2.U((loc==i),2),marker{i});
%     scatter(model_dpp.U((loc==i),1),model_dpp.U((loc==i),2),30,marker(i,:),'filled');
    scatter3(model_dpp.U((loc==i),1),model_dpp.U((loc==i),2),model_dpp.U((loc==i),3),30,marker(i,:),'filled');

%     scatter3(model2.U((loc==i),1),model2.U((loc==i),2),model2.U((loc==i),3), 10,marker(i,:));
end
view(40,50)
saveas(gcf,strcat(data_dir,strcat(int2str(rank),'_DPPculstering.png')))



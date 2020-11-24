
clear; close all; clc;

% choose which pre-trained model that we want to use in cnn
net = alexnet;

% apparently for the alexnet or any other seriesnetwork object, 
% we only need the net.Layers to train the model
layers = net.Layers;
    
% modify the layer
layers(23) = fullyConnectedLayer(3);
layers(25)=classificationLayer;

% read the data
allImages = imageDatastore('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Assignment\Lab1\SelectedDataSet', 'Includesubfolders', true, 'LabelSource', 'foldernames'); 
allImages.ReadFcn = @customReadDatastoreImage;

% classify the amount of train and test dataset
[trainingImages, testImages]=splitEachLabel(allImages, 0.8, 'randomize');

%%
%check the image size after resize
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end


%%
% specify the training options
opts=trainingOptions('sgdm','InitialLearnrate', 0.001, 'MaxEpochs', 5, 'MiniBatchSize', 20, 'Plots','training-progress');

% train the datasets 
myNet=trainNetwork(trainingImages, layers, opts);

%classify validation images
predictedLabels = classify (myNet, testImages);

% calculate the classification accuaracy
accuracy = mean(predictedLabels == testImages.Labels);


% sample of data resize function
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[227 227]);
end

clear; close all; clc;

% choose which pre-trained model that we want to use in cnn
net = googlenet;

% read the data
allImages = imageDatastore('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Assignment\Lab1\SelectedDataSet', 'Includesubfolders', true, 'LabelSource', 'foldernames'); 
allImages.ReadFcn = @customReadDatastoreImage;

% classify the amount of train and test dataset
[trainingImages, testImages]=splitEachLabel(allImages, 0.8, 'randomize');

% modify the layer
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
numClasses = numel(categories(trainingImages.Labels));

newLearnableLayer = fullyConnectedLayer(3, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

%%
% check the graph or the structure of the layer after modified
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%%
% check the image size after resize
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
myNet=trainNetwork(trainingImages, lgraph, opts);

%classify validation images
predictedLabels = classify (myNet, testImages);

% calculate the classification accuracy
accuracy = mean(predictedLabels == testImages.Labels);


% sample of data resize function
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end







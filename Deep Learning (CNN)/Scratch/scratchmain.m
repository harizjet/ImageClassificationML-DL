
clear; close all; clc;

% read the data
allImages = imageDatastore('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Assignment\Lab1\SelectedDataSet', 'Includesubfolders', true, 'LabelSource', 'foldernames'); 
allImages.ReadFcn = @customReadDatastoreImage;

% classify the amount of train and test dataset
[trainingImages, testImages]=splitEachLabel(allImages, 0.8, 'randomize');

% design the layer
layers = [
    imageInputLayer([28 28 3])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2) % note that we also can use average pooling 

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2) % note that we also can use average pooling

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];


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
% specify the options to train
% note that we can also change all this option
options = trainingOptions('sgdm', ... 
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testImages, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train the data
myNet = trainNetwork(trainingImages,layers,options);

% find the accuracy
YPred = classify(myNet,testImages);
YValidation = testImages.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

% sample of data resize function
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[28 28]);
end
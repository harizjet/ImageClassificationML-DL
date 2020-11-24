clear; close all; clc;
% Histogram of Gradient
% The goal is to create the input data that can be run in matlab

% get the location of the climbing set of the data
files = dir('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\STANFORDRESIZE50activities\Climbing\*.jpg');
filename = {files(:).name};
filelocation = {files(:).folder};
% get the location of the riding horse set of the data
files1 = dir('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\STANFORDRESIZE50activities\riding horse\*.jpg');
filename1 = {files1(:).name};
filelocation1 = {files1(:).folder};
% get the location of the running set of the data
files2 = dir('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\STANFORDRESIZE50activities\running\*.jpg');
filename2 = {files2(:).name};
filelocation2 = {files2(:).folder};

current_pos = 0;

% extract the data from the climbing location
for n=1:length(filename)
     X = imread([filelocation{n} '\' filename{n}]);        	 % read an image
     K=imresize(X, [20 20]);        	 %  resize the image, also note that image size also could affect feature extraction
     %I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = extractHOGFeatures(K);	%  extract HOG feature
     HH(n + current_pos,:) = features;		 % store HOG feature data into column matrix

     % create output for first data set
     output(n + current_pos,:) = 1;
     output(n + current_pos,2:2) = 0;
     output(n + current_pos,3:3) = 0;
end

current_pos = current_pos + length(filename);

% extract the data from the riding horse location
for n=1:length(filename1)
    X = imread([filelocation1{n} '\' filename1{n}]);        	 % read an image
     K=imresize(X, [20 20]);        	 %  resize the image, also note that image size also could affect feature extraction
     %I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = extractHOGFeatures(K);	%  extract HOG feature
     HH(n + current_pos,:) = features;		 % store HOG feature data into column matrix

     % create output for second data set
     output(n + current_pos,:) = 0;
     output(n + current_pos,2:2) = 1;
     output(n + current_pos,3:3) = 0;
end

current_pos = current_pos + length(filename1);

% extract the data from the running location
for n=1:length(filename2)
    X = imread([filelocation2{n} '\' filename2{n}]);        	 % read an image
     K=imresize(X, [20 20]);        	 %  resize the image, also note that image size also could affect feature extraction
     %I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = extractHOGFeatures(K);	%  extract HOG feature
     HH(n + current_pos,:) = features;		 % store HOG feature data into column matrix

     % create output for second data set
     output(n + current_pos,:) = 0;
     output(n + current_pos,2:2) = 0;
     output(n + current_pos,3:3) = 1;
end

current_pos = current_pos + length(filename2);

% write the input data into excel
T = array2table(HH);
filename = 'HOGtrain_input.xlsx';
writetable(T,filename, 'WriteVariableNames',0)

% write the output data into excel
O = array2table(output);
filename = 'HOGtrain_output.xlsx';
writetable(O,filename, 'WriteVariableNames',0)



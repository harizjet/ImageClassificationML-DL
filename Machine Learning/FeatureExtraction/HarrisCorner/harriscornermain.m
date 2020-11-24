clear; close all; clc;
% Harris Corner
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
     K=imresize(X, [100 100]);        	 %  resize the image, also note that image size also could affect feature extraction
     I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = detectHarrisFeatures(I);	%  extract HarrisCorner feature
     [tempfeatures, valid_corners] = extractFeatures(I, features);
     
     tempsingle = [];
     for i=1:height(table(valid_corners.Location))    % loops over every row in the table of the valid corner
         temppoint = valid_corners.Location(i, 1:2);    % get the (x, y) position of the valid corner
         euclid_dist = sqrt((temppoint(1)-0)^2 + (temppoint(2)-0)^2);   % calculate the euclidean distance between the corner point and 0,0
         tempsingle(i) = euclid_dist;   % append the euclidean distance into an array
     end
     
     HH(n + current_pos,1:length(tempsingle)) = tempsingle;     % store HarrisCorner feature data into column matrix
     
     % create output for first data set
     output(n + current_pos,:) = 1; 
     output(n + current_pos,2:2) = 0;
     output(n + current_pos,3:3) = 0;
end

current_pos = current_pos + length(filename);

% extract the data from the riding horse location
for n=1:length(filename1)
    X = imread([filelocation1{n} '\' filename1{n}]);        	 % read an image
     K=imresize(X, [100 100]);        	 %  resize the image, also note that image size also could affect feature extraction
     I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = detectHarrisFeatures(I);	%  extract HarrisCorner feature
     [tempfeatures, valid_corners] = extractFeatures(I, features);
     
     tempsingle = [];
     for i=1:height(table(valid_corners.Location))    % loops over every row in the table of the valid corner
         temppoint = valid_corners.Location(i, 1:2);    % get the (x, y) position of the valid corner
         euclid_dist = sqrt((temppoint(1)-0)^2 + (temppoint(2)-0)^2);   % calculate the euclidean distance between the corner point and 0,0
         tempsingle(i) = euclid_dist;   % append the euclidean distance into an array
     end
     
     HH(n + current_pos,1:length(tempsingle)) = tempsingle;     % store HarrisCorner feature data into column matrix
     
     % create output for second data set
     output(n + current_pos,:) = 0;
     output(n + current_pos,2:2) = 1;
     output(n + current_pos,3:3) = 0;
end

current_pos = current_pos + length(filename1);

% extract the data from the running location
for n=1:length(filename2)
    X = imread([filelocation2{n} '\' filename2{n}]);        	 % read an image
     K=imresize(X, [100 100]);        	 %  resize the image, also note that image size also could affect feature extraction
     I=rgb2gray(K);    		   	 % convert the colour image into gray scale
     features = detectHarrisFeatures(I);	%  extract HarrisCorner feature
     [tempfeatures, valid_corners] = extractFeatures(I, features);
     
     tempsingle = [];
     for i=1:height(table(valid_corners.Location))    % loops over every row in the table of the valid corner
         temppoint = valid_corners.Location(i, 1:2);    % get the (x, y) position of the valid corner
         euclid_dist = sqrt((temppoint(1)-0)^2 + (temppoint(2)-0)^2);   % calculate the euclidean distance between the corner point and 0,0
         tempsingle(i) = euclid_dist;   % append the euclidean distance into an array
     end
     
     HH(n + current_pos,1:length(tempsingle)) = tempsingle;     % store HarrisCorner feature data into column matrix
   
     % create output for second data set
     output(n + current_pos,:) = 0;
     output(n + current_pos,2:2) = 0;
     output(n + current_pos,3:3) = 1;
end

current_pos = current_pos + length(filename2);

% write the input data into excel
T = array2table(HH);
filename = 'HarrisCornertrain_input.xlsx';
writetable(T,filename, 'WriteVariableNames',0)

% write the output data into excel
O = array2table(output);
filename = 'HarrisCornertrain_output.xlsx';
writetable(O,filename, 'WriteVariableNames',0)




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import the data
surfdf = pd.read_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\Surf\output_input\SURFtrain_input_output.xlsx', header = None)
lbpdf = pd.read_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\LBP\output_input\LBPtrain_input_output.xlsx', header = None)
hogdf = pd.read_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\HOG\output_input\HOGtrain_input_output.xlsx', header = None)
harriscornerdf = pd.read_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\HarrisCorner\output_input\HarrisCornertrain_input_output.xlsx', header = None)

#split the data into output and input
surfoutput, surfinput = surfdf.iloc[:, 54:55], surfdf.iloc[:,:49]
lbpoutput, lbpinput = lbpdf.iloc[:, 63:64], lbpdf.iloc[:,:59]
hogoutput, hoginput = hogdf.iloc[:, 40:41], hogdf.iloc[:,:36]
harriscorneroutput, harriscornerinput = harriscornerdf.iloc[:, 81:82], harriscornerdf.iloc[:,:77]

X_train_surf, X_test_surf, y_train_surf, y_test_surf = train_test_split(surfinput, surfoutput, test_size = 0.3)
X_train_lbp, X_test_lbp, y_train_lbp, y_test_lbp = train_test_split(lbpinput, lbpoutput, test_size = 0.3)
X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(hoginput, hogoutput, test_size = 0.3)
X_train_harriscorner, X_test_harriscorner, y_train_harriscorner, y_test_harriscorner = train_test_split(harriscornerinput, harriscorneroutput, test_size = 0.3)

#decide the number of neighbors of the data
surf_knn = MLPClassifier(solver = 'adam', hidden_layer_sizes=(int(len(surfdf.columns) / 2)), random_state=1)
lbp_knn = MLPClassifier(solver = 'adam', hidden_layer_sizes=(int(len(lbpdf.columns) / 2)), random_state=1)
hog_knn = MLPClassifier(solver = 'adam', hidden_layer_sizes=(int(len(hogdf.columns) / 2)), random_state=1)
harriscorner_knn = MLPClassifier(solver = 'adam', hidden_layer_sizes=(int(len(harriscornerdf.columns) / 2)), random_state=1)

#fit the model
surf_knn.fit(X_train_surf, y_train_surf.values.ravel())
lbp_knn.fit(X_train_lbp, y_train_lbp.values.ravel())
hog_knn.fit(X_train_hog, y_train_hog.values.ravel())
harriscorner_knn.fit(X_train_harriscorner, y_train_harriscorner.values.ravel())

#making predictions from the model
y_pred_surf = surf_knn.predict(X_test_surf)
y_pred_lbp = lbp_knn.predict(X_test_lbp)
y_pred_hog = hog_knn.predict(X_test_hog)
y_pred_harriscorner = harriscorner_knn.predict(X_test_harriscorner)

#check the accuracy
surf_accuracy = accuracy_score(y_test_surf, y_pred_surf)
lbp_accuracy = accuracy_score(y_test_lbp, y_pred_lbp)
hog_accuracy = accuracy_score(y_test_hog, y_pred_hog)
harriscorner_accuracy = accuracy_score(y_test_harriscorner, y_pred_harriscorner)

#output confusion matrix and classification report
# surfconfusion_matrix = confusion_matrix(y_test, y_pred)
surf_classification_report = classification_report(y_test_surf, y_pred_surf)
lbp_classification_report = classification_report(y_test_lbp, y_pred_lbp)
hog_classification_report = classification_report(y_test_hog, y_pred_hog)
harriscorner_classification_report = classification_report(y_test_harriscorner, y_pred_harriscorner)

print(f'Surf : {surf_accuracy}')
print(f'LBP : {lbp_accuracy}')
print(f'HOG : {hog_accuracy}')
print(f'HarrisCorner : {harriscorner_accuracy}')

print('----------------Surf Classification Report----------------')
print(surf_classification_report)
print('----------------------------------------------------------\n\n')
print('----------------LBP Classification Report----------------')
print(lbp_classification_report)
print('----------------------------------------------------------\n\n')
print('----------------HOG Classification Report----------------')
print(hog_classification_report)
print('----------------------------------------------------------\n\n')
print('----------------HarrisCorner Classification Report----------------')
print(harriscorner_classification_report)
print('----------------------------------------------------------\n\n')

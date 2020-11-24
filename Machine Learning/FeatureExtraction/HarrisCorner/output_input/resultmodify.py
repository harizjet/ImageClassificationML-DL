

import pandas as pd

df = pd.read_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\HarrisCorner\output_input\HarrisCornertrain_input_output1.xlsx', header = None)

#create a temp list
result = list()
#loops over every row and convert the result into one row
for i in range(len(df.index)):
    count = 0
    while list(df.iloc[i:i+1, 78:81].values)[0][count] != 1:
        count += 1
    result.append(count)

#transfer the result into the dataframe
result = pd.DataFrame(result)
df[82] = result

df.to_excel('D:\Academics\BeSpoke UiTM\CSC728 - Machine Learning\Projects\Lab1\Machine Learning\FeatureExtraction\HarrisCorner\output_input\HarrisCornertrain_input_output.xlsx', index=False, header=False)

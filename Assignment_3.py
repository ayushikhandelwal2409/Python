import pandas as pd
import numpy as np


# task 1
# ques 1
data = [25, 30, 35, 40, 45] 
ser=pd.Series(data)
print(ser)

# 0    25
# 1    30
# 2    35
# 3    40
# 4    45
# dtype: int64

# ques 2
ser=pd.Series(data,index=['A', 'B', 'C', 'D', 'E'])
print(ser)
# A    25
# B    30
# C    35
# D    40
# E    45
# dtype: int64

print(ser.head(3))
# A    25
# B    30
# C    35
# dtype: int64

print(ser.mean())
print(ser.median())
print(ser.std())
# 35.0
# 35.0
# 7.905694150420948

# task 2
# ques 1
data = {
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
}
df = pd.DataFrame(data)

# ques 2
print(df.head(2))
print("\nColumn names and data types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())
# Add a new column 'Passed'
df['Passed'] = df['Marks'] >= 80
print("\nDataFrame with 'Passed' column:")
print(df)

#     Name  Age  Gender  Marks
# 0  Alice   20  Female     85
# 1    Bob   22    Male     78

# Column names and data types:
# Name      object
# Age        int64
# Gender    object
# Marks      int64
# dtype: object

# Summary statistics:
#              Age      Marks
# count   5.000000   5.000000
# mean   20.400000  83.400000
# std     1.140175   7.334848
# min    19.000000  74.000000
# 25%    20.000000  78.000000
# 50%    20.000000  85.000000
# 75%    21.000000  88.000000
# max    22.000000  92.000000

# DataFrame with 'Passed' column:
#     Name  Age  Gender  Marks  Passed
# 0  Alice   20  Female     85    True
# 1    Bob   22    Male     78   False
# 2  Carol   19  Female     92    True
# 3  David   21    Male     74   False
# 4    Eve   20  Female     88    True

# task 3
# ques 1

print(df[['Name','Marks']])
#     Name  Marks
# 0  Alice     85
# 1    Bob     78
# 2  Carol     92
# 3  David     74
# 4    Eve     88

print(df[df['Marks']>80])
#     Name  Age  Gender  Marks  Passed
# 0  Alice   20  Female     85    True
# 2  Carol   19  Female     92    True
# 4    Eve   20  Female     88    True

print(df.loc[df['Marks'].idxmax()])
# Name       Carol
# Age           19
# Gender    Female
# Marks         92
# Passed      True
# Name: 2, dtype: object

# task 4
# ques 1,2,3

df.loc[1,'Marks']=None
df.loc[4,'Age']=None
newdf=pd.DataFrame(df)
print(df)
#     Name   Age  Gender  Marks  Passed
# 0  Alice  20.0  Female   85.0    True
# 1    Bob  22.0    Male    NaN   False
# 2  Carol  19.0  Female   92.0    True
# 3  David  21.0    Male   74.0   False
# 4    Eve   NaN  Female   88.0    True

# ques 4
print(df.isnull().sum())
# Name      0
# Age       1
# Gender    0
# Marks     1
# Passed    0
# dtype: int64

df['Marks'].fillna(df['Marks'].mean(),inplace=True)
df.dropna(subset=['Age'],inplace=True)
print(df)
#     Name   Age  Gender  Marks  Passed
# 0  Alice  20.0  Female  85.00    True
# 1    Bob  22.0    Male  84.75   False
# 2  Carol  19.0  Female  92.00    True
# 3  David  21.0    Male  74.00   False

# task 5
data = {
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
    'Age': [20, 22, 19, 21, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'Marks': [85, 78, 92, 74, 88]
}
df=pd.DataFrame(data)
# ques1
print(df.groupby('Gender').agg({'Age': 'mean', 'Marks': 'mean'}))

#               Age      Marks
# Gender                      
# Female  19.666667  88.333333
# Male    21.500000  76.000000

print(df['Gender'].value_counts())
# Gender
# Female    3
# Male      2
# Name: count, dtype: int64

# task 6
# ques 1

newdf.to_csv('students_data.csv',index=False)
dfs=pd.read_csv('students_data.csv')
print(dfs.head())
#     Name   Age  Gender  Marks  Passed
# 0  Alice  20.0  Female  85.00    True
# 1    Bob  22.0    Male  84.75   False
# 2  Carol  19.0  Female  92.00    True
# 3  David  21.0    Male  74.00   False
# 4    Eve   NaN  Female  88.00    True


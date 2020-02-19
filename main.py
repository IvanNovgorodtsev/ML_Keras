import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("train_2v.csv")
print(df.describe())
df = df.drop('id', 1)

df.ever_married = pd.Series(np.where(df.ever_married == 'Yes', 1, 0), df.index)
df.gender = pd.Series(np.where(df.gender == 'Male', 1, 0), df.index)

print(df.work_type.count)
print(df.work_type.unique())
print(df.Residence_type.unique())
print(df.smoking_status.unique())

print(df.columns)
print(df.ever_married.describe())

print(df.corr())
# plt.matshow(df.corr())
# plt.show()

dff = pd.get_dummies(data=df, columns=['work_type', 'Residence_type','smoking_status'])
print(dff.columns)


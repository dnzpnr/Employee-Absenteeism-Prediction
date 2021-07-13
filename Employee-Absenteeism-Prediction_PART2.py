df2.head()

df2['Absenteeism Time in Hours'].median()

targets = np.where(df2['Absenteeism Time in Hours'] > 3,1,0)
targets

df2['New_Target'] = targets
df2.head()

df2['New_Target'].value_counts()

df2.drop(['Absenteeism Time in Hours','Date'],axis=1,inplace=True)
df2.head()

df2.head()

# Simdi model kurmaya baslayabiliriz.
x = df2.drop(['New_Target'], axis=1)
y = df2['New_Target']

# Verisetimizi standartlastiralim
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
x_scaled.shape

# Şimdi verisetini train ve test seti olarak ayıralım;
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.8, random_state=40)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
reg = LogisticRegression()
reg.fit(x_train,y_train)
reg.score(x_train,y_train)

reg = LogisticRegression()
reg.fit(x_train,y_train)

reg.score(x_train,y_train)

reg.intercept_

reg.coef_

features = x.columns.values
features

summary_table = pd.DataFrame(columns=['Feature_Names'], data= features)
summary_table['coefs'] = np.transpose(reg.coef_)
summary_table

summary_table.index += 1
summary_table

summary_table.loc[0] = ['intercept',reg.intercept_[0]]
summary_table

summary_table = summary_table.sort_index()
summary_table

summary_table['exp_coefs'] = np.exp(summary_table.coefs)
summary_table

summary_table.sort_values('exp_coefs', ascending=False)

x1 = df2.drop(['New_Target','Body Mass Index','Month','Education'], axis=1)
x1_scaled = scaler.fit_transform(x1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1_scaled,y, test_size=0.8, random_state=40)
reg.fit(x1_train,y1_train)
reg.score(x1_train,y1_train)

reg.score(x1_train,y1_train)

reg.score(x1_test,y1_test)

reg.predict_proba(x1_test)

import pickle
with open('model','wb') as file:
    pickle.dump(reg,file)

with open('scaler','wb') as file:
    pickle.dump(scaler,file)

raw_data = pd.read_csv('/kaggle/input/employee-absenteeism-prediction/Absenteeism-data.csv')
raw_data

df = raw_data.copy()
df.head()

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df.info()

df['ID'].value_counts()

df.drop(['ID'], axis=1, inplace=True)
df.head()

df['Reason for Absence'].unique()

len(df['Reason for Absence'].unique())

df['Reason for Absence'].min()

df['Reason for Absence'].max()

sorted(df['Reason for Absence'].unique())

reason_cols = pd.get_dummies(df['Reason for Absence'])
reason_cols.head()

reason_cols['check'] = reason_cols.sum(axis=1)
reason_cols.head(10)

reason_cols['check'].value_counts()

reason_cols.drop(['check', 0],axis=1, inplace=True)
reason_cols.head()

df.drop('Reason for Absence', axis=1,inplace=True)
df.head()

reason_group1 = reason_cols.loc[:,1:14].max(axis=1)
reason_group2 = reason_cols.loc[:,15:17].max(axis=1)
reason_group3 = reason_cols.loc[:,18:21].max(axis=1)
reason_group4 = reason_cols.loc[:,22:].max(axis=1)

reason_group1.head()

df = pd.concat([df,reason_group1,reason_group2,reason_group3,reason_group4], axis=1)
df.head()

df.columns.values

new_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Group1', 'Group2','Group3','Group4']
df.columns = new_names
df.head()

ordered_names = [ 'Group1', 'Group2','Group3','Group4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[ordered_names]
df.head()

# CHECKPOINT 1
df1 = df.copy()
df1.head()

df1.info()

df1['Date'] = pd.to_datetime(df1['Date'],format="%d/%m/%Y")

df1['Date'][0]

df1['Date'][0].month

month_list = []

for i in range(len(df1['Date'])):
    month_list.append(df1['Date'][i].month)

len(month_list)

df1['Month'] = month_list
df1.head()

df1['Date'][0].weekday()

def weekday(date):
    return date.weekday()

df1['Day'] = df1['Date'].apply(weekday)
df1.head(10)

df1['Education'].unique()

df1['Education'].value_counts()

df1['Education'] = df1['Education'].map({1:0,2:1,3:1,4:1})
df1['Education'].value_counts()

# CHECKPOINT 2
df2 = df1.copy()
df2.head()

df2.head()

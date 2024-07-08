import pandas as pd

df=pd.read_csv('USA_Housing.csv')

print(df['Price'])

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
import pandas as pd

df = pd.read_csv('api_test/db_setup/books.csv')

df = df.fillna('other')

print(df.isnull().sum())
#print(df.head(5))

df.to_csv('api_test/db_setup/books2.csv')





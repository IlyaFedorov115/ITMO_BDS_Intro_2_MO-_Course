import pandas as pd
df = pd.read_csv('https://courses.openedu.ru/asset-v1:ITMOUniversity+DATANFST2035+cifru_2035+type@asset+block@pulsar_stars_new.csv')

print(df.shape)
print(df.info())

df.describe()
df[((df['TG']==0) & ((df['MIP']>=95.8984375) & (df['MIP'] <= 96.4140625))) | ((df['TG']==1) & ((df['MIP']>=77.4921875) & (df['MIP']<=83.7734375)))].count()
new_df = df[((df['TG']==0) & ((df['MIP']>=95.8984375) & (df['MIP'] <= 96.4140625))) | ((df['TG']==1) & ((df['MIP']>=77.4921875) & (df['MIP']<=83.7734375)))]
new_df['MIP'].mean()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Coding journey\Projects\Visualization projects\Top Goalscorers for Country\goals.csv')

df = df.sort_values(['Nation','Goals'],ascending=[True,False])
#print(df.head(10))

nations = df['Nation'].unique()
for nation in nations:
    data = df[df['Nation']==nation]
    plt.figure(figsize=(6,4))
    plt.bar(data['Player'],data['Goals'])
    plt.title(f"Top 3 Goal Scorers - {nation}")
    plt.xlabel("Player")
    plt.show()
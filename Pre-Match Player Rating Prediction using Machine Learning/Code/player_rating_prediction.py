import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('ratings.csv')

df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].str.strip()

#Convert date to datetime format and sort by date and player    
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce",format="%d/%m/%y")
df = df.sort_values(by=['Date'])

# Convert 'Rating' and 'OpponentRank' to numeric, handle errors, and create new features
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=['Rating'])
df["OpponentRank"] = pd.to_numeric(df["OpponentRank"], errors="coerce")

#Create column 'Form' as the rolling average of the last 5 ratings for each player
df['Form'] = (
    df.groupby('Player')['Rating']
      .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
df['Form'] = df['Form'].fillna(df['Form'].median())

#Fill missing 'OpponentRank' with a value higher than the maximum rank in the dataset
max_rank = df["OpponentRank"].max(skipna=True)
fill_value = int(max_rank) + 10
df["OpponentRank"] = df["OpponentRank"].fillna(fill_value)

#Calculate rest days between matches for each player
df['RestDays'] = df.groupby('Player')['Date'].diff().dt.days

#Calculate opponent strength as a normalized value based on opponent rank
max_rank = df["OpponentRank"].max()
min_rank = df["OpponentRank"].min()
df["OpponentStrength"] = 1 - (
    (df["OpponentRank"] - min_rank) /
    (max_rank - min_rank)
)

#Fill missing 'RestDays' with the median value
df['RestDays'] = df['RestDays'].fillna(df['RestDays'].median())

#Create a binary feature for each player for their historical ratings
df["PlayerBaseline"] = (
    df.groupby("Player")["Rating"]
      .transform(lambda x: x.shift(1).expanding().mean())
)
df['PlayerBaseline'] = df['PlayerBaseline'].fillna(df['PlayerBaseline'].median())

#One-hot encode the 'Position' column, dropping the first category to avoid multicollinearity
df = pd.get_dummies(df, columns=["Position"], drop_first=True)

print(df.head())

X = df.drop(columns=["Player", "Date", "Rating", "Team", "Opponent", "OpponentRank"],axis=1)
y = df.Rating

split_index = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

xg = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

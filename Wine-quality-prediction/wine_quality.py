import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load datasets
rdf = pd.read_csv(r'D:\Coding journey\Projects\Wine-quality-prediction\dataset\winequality-red.csv', sep=';')
wdf = pd.read_csv(r'D:\Coding journey\Projects\Wine-quality-prediction\dataset\winequality-white.csv', sep=';')

# Add wine type column
rdf['type'] = 0   # red
wdf['type'] = 1   # white

# Function to create quality labels
def label_quality(q):
    if q < 5:
        return "Bad"
    elif q <= 6:
        return "Average"
    else:
        return "Good"

rdf['quality_label'] = rdf['quality'].apply(label_quality)
wdf['quality_label'] = wdf['quality'].apply(label_quality)

# Combine datasets (IMPORTANT for model)
df = pd.concat([rdf, wdf], ignore_index=True)

# ---------- FUNCTION FOR SUMMARY ----------
def wine_summary(data, name):
    total = data.shape[0]

    above_6 = data[data['quality'] > 6].shape[0]
    below_5 = data[data['quality'] < 5].shape[0]
    between = data[(data['quality'] >= 5) & (data['quality'] <= 6)].shape[0]

    percent_good = (above_6 * 100) / total

    print(f"\n{name} WINE SUMMARY")
    print("-" * 30)
    print(f"Total samples: {total}")
    print(f"Quality > 6: {above_6}")
    print(f"Quality < 5: {below_5}")
    print(f"Quality 5–6: {between}")
    print(f"Good quality %: {percent_good:.2f}%")

    print("\nSkewness:")
    print(data.skew(numeric_only=True))


# Call function
wine_summary(rdf, "RED")
wine_summary(wdf, "WHITE")


# ---------- VISUALIZATION ----------

# Quality distribution
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df)
plt.title("Overall Wine Quality Distribution")
plt.show()


# Compare red vs white quality
plt.figure(figsize=(6,4))
sns.countplot(x='quality', hue='type', data=df)
plt.title("Quality: Red vs White")
plt.legend(labels=['Red', 'White'])
plt.show()


# Correlation heatmap (important for ML)
plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()


# ---------- OPTIONAL: SAVE CLEAN DATA ----------
df.to_csv("cleaned_wine_data.csv", index=False)

X = df.drop(['quality','quality_label'],axis=1)
y = df['quality_label']

y = y.map({'Bad':0, 'Average':1, 'Good':2})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

rf =RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

xgb = XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
xgb.fit(X_train,y_train)

y_pred_xgb =  xgb.predict(X_test)

rf_params = {
    'n_estimators': [100,200,300],
    'max_depth': [5,10,15,None],
    'min_samples_split': [2,5,10]
    }

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),rf_params,cv=5)
rf_grid.fit(X_train,y_train)
rf = rf_grid.best_estimator_

def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} RESULTS")
    print("-" * 30)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

evaluate_model('Logistic Regression', y_test, y_pred_lr)
evaluate_model('Random Forest', y_test, y_pred_rf)
evaluate_model('XG Boost', y_test, y_pred_xgb)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

models = ['Logistic', 'Random Forest', 'XGBoost']
accuracies = [acc_lr, acc_rf, acc_xgb]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

joblib.dump(xgb, "best_model.pkl")   # or rf if better
joblib.dump(scaler, "scaler.pkl")
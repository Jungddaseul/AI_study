import pickle   # 파일 만들기
import sklearn as sk  # ML

from sklearn.model_selection import train_test_split  # 데이터 나누기 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer  # 유방암 데이터 셋 불러오기 
import pandas as pd

#  데이터 불러오기 
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(cancer_df.columns)

# 피처와 레이블를 지정.
sel = ['worst texture', 'worst concave points', 'mean texture', 'area error', 'worst perimeter']
X = cancer_df[sel]
y = cancer.target

# 데이터 나누기
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 지정 및 학습 
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

print("랜덤 포레스트 모델 정확도 : ", model_rf.score(X_test, y_test))
pickle.dump(model_rf, open('cancer_rf.pkl', 'wb'))
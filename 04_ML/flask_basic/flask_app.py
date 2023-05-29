from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# 모델 가져오기
print(os.getcwd())  # 현재 작업 경로 확인
path = "C:\\Users\\161332\\Documents\\github\AI_study\\04_ML\\flask_basic"  # pkl 파일 경로 확인
os.chdir(path)  # 현재 작업 경로를 변경
print(os.getcwd())  # 현재 작업 경로 확인
model = pickle.load(open("cancer_rf.pkl", 'rb'))  # 작업 경로에서 파일 가져오기
print(model)

# 플라스크 앱 객체 지정
app = Flask(__name__)

# 플라스크 앱의 루트 디렉터리를 초기화
@app.route('/')

def main():
    return render_template('start.html')

# 04 초기 웹 페이지에서 submit 했을 때 실행
# request.form['']을 사용하여 HTML 페이지에서 입력한 데이터를 가져온다.
# model.predict()를 통해 클래스를 예측한다.
# 예측값에 따라 어떤 텍스트와 이미지를 보낼지, after.html에 설정.

@app.route('/predict', methods=['POST'])
def home():
# html의 입력한 값을 넘겨받는다.
    val1 = request.form['a']
    val2 = request.form['b']
    val3 = request.form['c']
    val4 = request.form['d']
    val5 = request.form['e']
    val1 = float(val1); val2 = float(val2);
    val3 = float(val3); val4 = float(val4); val5 = float(val5)

    arr = np.array([[val1, val2, val3, val4, val5]])
    pred = model.predict(arr)

    # 렌더링할 html 파일명, 전달할 변수
    return render_template('after.html', data=pred)

# 05 직접 실행된 경우, 앱을 디버그 on 모드로 실행
if __name__ == "__main__":
   app.run(debug=True)
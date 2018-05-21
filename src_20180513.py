# 以下のデータ形式を前提としたプログラムです。
# ---------データ形式---------
# ・ファイル名
# 　・モデル用：final_hr_analysis_train.csv
# 　・スコア用：final_hr_analysis_test.csv
# ・データ形式
# 　・第1列：index（ID列）
# 　・第2列：left（1：退職、0：非退職の正解ラベル）
# 　・第3列以降：特徴量
# 　・カテゴリ変数名：sales, salary
# 　・ヘッダー項目（以下11項目、以下順番で構成）
# 　　・index
# 　　・left
# 　　・satisfaction_level
# 　　・last_evaluation
# 　　・number_project
# 　　・average_montly_hours
# 　　・time_spend_company
# 　　・Work_accident
# 　　・promotion_last_5years
# 　　・sales
# 　　・salary

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.datasets import load_iris
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFE  
# models
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# evaluation indexes
from sklearn.metrics import f1_score, accuracy_score

#モードの選択(学習:train、評価:test)
mode = input('train or test >>>')
#CSVファイル形式のデータの読み込み
filename = input('csv file name >>>')
data = pd.read_csv('./data/str(filename), header=0)

##カテゴリデータの列名の取得
COLS_OHE = []
while(True):
    input_cate = input('Input the name of categorical columns >>> ')
    if input_cate != '' and input_cate != '/end':
        COLS_OHE.append(input_cate)
    elif input_cate == '/end':
        break
    else:
        print('invalid input')
        continue
        
#ID、クラス変数、特徴量にデータを分割
##
if mode != 'test':
    ID = data.iloc[:, [0]]
    CLASS = data.iloc[:, [1]]
    X = data.iloc[:, 2:]
else:
    ID = data.iloc[:, [0]]
    X = data.iloc[:, 1:]
    #X.drop("left", axis=1)
    del X['left']
    
#前処理
## one-hot encoding X
X_ohe = pd.get_dummies(X, dummy_na=True, columns=COLS_OHE)

## 欠損処理 Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_ohe)

X_imp = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe.columns.values)
if mode != 'test':
    imp.fit(CLASS)
    CLASS = pd.DataFrame(imp.transform(CLASS), columns=CLASS.columns.values)

if mode == 'train':#train-->>
    
    ## 特徴量選択 RFE
    selector = RFE(estimator=GradientBoostingRegressor(random_state=0),
                   n_features_to_select=20,
                   step=0.05)
    selector.fit(X_imp,
                 CLASS.as_matrix().ravel())
    X_fin = pd.DataFrame(selector.transform(X_imp),
                         columns=X_imp.columns.values[selector.support_])

    #評価指標の選択（入力受付）
    print('\n############################')
    print('# select evaluation index')
    print('#   f1_score:        f')
    print('#   accuracy_score:  a')
    print('##############################')
    while(True):
        evaluation = input('select evaluation following help >>>')
        if evaluation == 'f' or evaluation == 'a':
            break
        else:
            print('Invalid input')
            continue
    ## hold-out
    X_train,X_test,y_train,y_test = train_test_split(X_fin, CLASS, test_size=0.20, random_state=5)

    y_train['left'] = y_train['left'].astype(np.int64)
    y_test['left'] = y_test['left'].astype(np.int64)

    # pipeline
    pipe_logistic = Pipeline([('scl',StandardScaler()),('est',LogisticRegression(random_state=1))])
    pipe_gbc = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier(random_state=1))])

    # パラメータグリッドの設定
    param_grid_logistic = {'est__C':[0.1,1.0,10.0,100.0], 'est__penalty':['l1','l2']}
    param_grid_gbc = {'est__n_estimators':[50,100],'est__subsample':[0.8, 1.0]}    
    
    # 学習
    pipes = [pipe_logistic, pipe_gbc]
    params = [param_grid_logistic, param_grid_gbc]
    names = ['logistic', 'gbc']
    results = {}
    best_estimator = []
    for pipe, param, name in zip(pipes, params, names):
        print('探索空間:%s' % param)
        gs = GridSearchCV(estimator=pipe, param_grid=param, scoring='f1', cv=3)
        gs = gs.fit(X_train, y_train.as_matrix().ravel())
        if evaluation == 'f':
            results[name] = f1_score(y_test.as_matrix().ravel(),gs.predict(X_test))
        elif evaluation == 'a':
            results[name] = accuracy_score(y_test.as_matrix().ravel(),gs.predict(X_test))
        best_estimator.append(gs.best_estimator_)   # gs.best_estimator_でベストモデルを呼び出せる
        print('Best Score %.6f\n' % gs.best_score_) # gs.best_score_で上記ベストモデルのCV評価値（ここではf1スコア）を呼び出せる
        print('Best Model: %s' % gs.best_estimator_)
    
    print('----------------------------------------------------------------------------------------------')
    print('### algorithm ranking ###')
    for key, value in sorted(results.items(), key=lambda x: -x[1]):
        print(str(key) + ": " + str(value))
    
    pipe_names = ['Logistic', 'GradientBoosting']
    for i, est in enumerate(best_estimator):
        joblib.dump(est, pipe_names[i] + '.pkl') 
    # only train --<<
    
#学習済みモデルの呼び出し&予測確率出力
elif mode == 'test':#predict-->>
    
    model_name = input('input the name of model >>>')    
    model = joblib.load(str(model_name) + '.pkl')
    
    #IDと予測値をcsv形式で出力
    y_pred = pd.DataFrame(model.predict(X_fin), columns=['y_pred'])
    y_proba = pd.DataFrame(model.predict_proba(X_fin)).iloc[:, [-1]]
    y_proba = y_proba.rename(columns={1 : '確率'}, inplace=False)
        
    df_con = pd.concat([ID, y_proba], axis=1, join='inner')
    df_con_2 = pd.concat([ID, y_pred, y_proba], axis=1, join='inner')
    
    #予測結果出力
    display(df_con_2)
    
    #CSV出力（1である確率が欲しいため、カラム「1」の予測確率を出力）
    df_con.to_csv("./data/result.csv", index=False)

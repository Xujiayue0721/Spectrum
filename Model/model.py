import joblib
from keras.losses import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, PredefinedSplit
import numpy as np
from sklearn.metrics import r2_score



def PLR(x_train,x_test,y_train,y_test):#PLS
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)
    # print(x_train)
    # print(y_train)
    pls_model_setup = PLSRegression(scale=True)
    param_grid = {'n_components': range(8,9)}
    # GridSearchCV优化参数、训练模型
    gsearch = GridSearchCV(pls_model_setup, param_grid, refit = True,cv=4)
    pls_model = gsearch.fit(x_train, y_train)
    # print(pls_model.cv_results_['mean_test_score'])
    arr_data=pls_model.best_estimator_.coef_
    print(arr_data)
    print(pls_model.best_estimator_.intercept_)
    print(pls_model.best_params_)
    # print(pls_model.cv_results_)
    xs_data=pls_model.best_estimator_.x_scores_
    # xs_data=xs_data.T
    plt.plot(xs_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("{0}得分".format('PLSR的'), fontsize='20')  # 添加标题
    plt.xlabel("Wavelength(nm)")
    plt.ylabel('杂质')
    plt.show()
    # save model
    joblib.dump(pls_model, 'saved_model/model1.pkl')
    # 预测
    pred = pls_model.predict(x_test)
    print('PLSR训练集上的MAE/MSE/决定系数')
    # print('Mean Absolute Error:',mean_absolute_error(y_test, pred))
    # print('Mean Squared Error:',mean_squared_error(y_test, pred))
    print('Root Mean Squared Error:',np.sqrt(mean_squared_error(pred,y_test)))
    d=np.average(y_test-pred)
    arr=(y_test-pred-d)*(y_test-pred-d)
    ans=0
    for i in range(len(arr)-1):
        ans+=arr[i]
    ans=float(abs(ans))
    sse=np.sqrt(ans/(len(pred)-1))
    print('SSE:', sse)
    sqrtn=np.sqrt(len(y_test))
    t=abs((abs(d)*sqrtn)/sse)
    print("t:",t)
    print('相关系数:',r2_score(y_test, pred))
    print('RPD:',1/(np.sqrt(1-r2_score(y_test, pred))))
    return y_test,pred

def Model(method, x_train,x_test,y_train,y_test):
    if method=="PLSR":
        y_test,pred=PLR(x_train,x_test,y_train,y_test)
    return y_test,pred

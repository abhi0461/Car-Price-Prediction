import pandas as pd
import seaborn as sns
from seaborn import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def main():
    dataset = pd.read_csv("car data.csv")
    print(dataset.head())
    print(dataset.shape)
    print(dataset["Transmission"].unique())
    print(dataset["Owner"].unique())
    # print(dataset.isnull().sum())
    print(dataset.describe())
    # print(dataset.info())
    print(dataset.columns)
    dataset.drop(["Car_Name"], axis=1, inplace= True)
    print(dataset.head())
    print(dataset.columns)
    dataset["Current_Year"] = 2020
    print(dataset.head())
    dataset["no_year"] = dataset["Current_Year"]-dataset["Year"]
    print(dataset.head())
    dataset.drop(["Year"],axis=1,inplace=True)
    dataset.drop(["Current_Year"],axis=1,inplace=True)
    print(dataset.head())
    dataset = pd.get_dummies(dataset,drop_first=True)
    print(dataset.head())
    print(dataset.corr())
    # sns.pairplot(dataset)
    # show()
    # corrmat = dataset.corr()
    # top_corr_features = corrmat.index
    # plt.figure(figsize=(20,20))
    # g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    # show()
    x = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]
    # print(x.head())
    # print(y.head())
    # model = ExtraTreesRegressor()
    # model.fit(x,y)
    # print(model.feature_importances_)

    # feat_importance = pd.Series(model.feature_importances_,index=x.columns)
    # feat_importance.nlargest(5).plot(kind='barh')
    # plt.show()

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    rf_random = RandomForestRegressor()

    #HyperParameters
    n_estimators = [int(x) for x in np.linspace(start = 100 , stop = 1200 ,num = 12 )]
    print(n_estimators)

    max_features = ["auto" , "sqrt"]
    max_depth = [int(x) for x in np.linspace(5,30 , num=6)]
    min_samples_split = [2,5,10,15,100]
    min_samples_leaf = [1,2,5,10]

    random_grid = {'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split' : min_samples_split,
    'min_samples_leaf': min_samples_leaf}
    print(random_grid)
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf , param_distributions = random_grid , scoring = 'neg_mean_squared_error' , n_iter = 10 ,cv =5 ,verbose =2 , random_state=42 , n_jobs = 1 )
    rf_random.fit(x_train , y_train)

    predictions=rf_random.predict(x_test)

    file = open("random_forest_regression_model.pkl","wb")
    pickle.dump(rf_random, file)

if __name__ == "__main__":
    main()
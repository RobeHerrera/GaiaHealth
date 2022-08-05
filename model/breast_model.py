import pandas as pd
# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Modeling
from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# Metrics
from sklearn.metrics import accuracy_score, f1_score


def run():
    cancer_data = pd.read_csv('data/data.csv')
    pd.options.display.max_columns = len(cancer_data)
    print(cancer_data.describe())
    print(f'Number of entries: {cancer_data.shape[0]:,}\n'
          f'Number of features: {cancer_data.shape[1]:,}\n\n'
          f'Number of missing values: {cancer_data.isnull().sum().sum()}\n\n')
    # print(cancer_data.head(2))

    # Clean Data
    # missing values per column
    cancer_data.isnull().sum()
    # How many women, in %, have a confirmed cancer (a malignant breast tumor)?
    (cancer_data['diagnosis'].value_counts(normalize=True) * 100).astype(int)

    # Let's drop the last column that contains only missing values:
    cancer_data = cancer_data.drop('Unnamed: 32', axis=1)
    # values
    X = cancer_data.iloc[:, 2:32].values

    # Results
    y = cancer_data.iloc[:, 1].values

    # Split data into train and test sets and scale the features
    RANDOM = 1

    # train (60%) validate (20%) test (20%)
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.4, random_state=RANDOM)
    (X_valid, X_test, y_valid, y_test) = train_test_split(X_valid, y_valid, test_size=0.5, random_state=RANDOM)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)
    # sc.inverse_transform(X_train)

    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_valid = labelencoder_y.transform(y_valid)
    y_test = labelencoder_y.transform(y_test)

    # Modeling
    # # K - nearest neighbors(KNN)
    # knn = KNeighborsClassifier()
    #
    # # defining parameter range
    # k_range = list(range(1, 31))
    # param_grid = dict(n_neighbors=k_range)
    # grid = GridSearchCV(knn, param_grid, cv=10, scoring='f1',
    #                     verbose=1, return_train_score=True)

    # fitting the model for grid search
    # grid_search = grid.fit(X_train, y_train)

    # # getting the results
    # best_params = grid_search.best_params_
    # best_f1score = grid_search.best_score_
    # print(f"Best params = {best_params}\n" +
    #       f"Best f1 score = {best_f1score}")
    #
    # # final model
    # knn = grid_search.best_estimator_
    #
    # # Logistic Regression
    lr = LogisticRegression()

    # defining parameter range
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [200, 300, 400, 500, 1000]
    }
    grid = GridSearchCV(lr, param_grid, cv=10, scoring='f1',
                        verbose=1, return_train_score=True)

    # fitting the model for grid search
    grid_search = grid.fit(X_train, y_train)

    # getting the results
    best_params = grid_search.best_params_
    best_f1score = grid_search.best_score_
    print(f"Best params = {best_params}\n" +
          f"Best f1 score = {best_f1score}")

    # final model
    lr = grid_search.best_estimator_

    # Train data
    lr.fit(X_train, y_train)
    lr_predictions = lr.predict(X_valid)

    print(f'Accuracy scores -> Logistic regression model: {accuracy_score(y_valid, lr_predictions):.3f}')

    print(f'f1 scores -> Logistic regression model: {f1_score(y_valid, lr_predictions):.3f}')

    final_model = lr
    predictions = final_model.predict(X_test)

    print(f'Final model:\n'
          f'Accuracy scores: {accuracy_score(y_test, predictions):.3f}\n'
          f'Accuracy scores: {f1_score(y_test, predictions):.3f}'
          )

    # Let's make some predictions
    print("Lets make some predictions...")
    # B = 0
    # M = 1
    index_to_search = 0
    new_data = [X[index_to_search]]
    print(new_data)
    print(y[index_to_search])
    new_data_transformed = sc.transform(new_data)
    new_prediction = final_model.predict(new_data_transformed)
    print(f"new prediction = {new_prediction}")

    print("My Test with CSV")
    my_test = pd.read_csv('data/testData.csv')
    new_data = my_test.iloc[:, :].values
    new_data_transformed = sc.transform(new_data)
    new_prediction = final_model.predict(new_data_transformed)
    print(f"new prediction CSV = {new_prediction}")

    # i = ''
    # while i != 'q':
    #     i = input("Enter Name, or 'q': ")
    #
    #     if i != 'q':
    #         index_to_search = int(i)
    #         new_data = [X[index_to_search]]
    #         print(new_data)
    #         print(y[index_to_search])
    #
    #         new_data_transformed = sc.transform(new_data)
    #         new_prediction = final_model.predict(new_data_transformed)
    #         print(f"new prediction = {new_prediction}")

    print("Model Finish")

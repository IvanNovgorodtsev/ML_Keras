import pandas as pd
import numpy as np
from keras import optimizers
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def visualization(df):
    print(df.work_type.count)
    print(df.work_type.unique())
    print(df.Residence_type.unique())
    print(df.smoking_status.unique())

    print(df.corr())
    plt.matshow(df.corr())
    plt.show()

    sns.catplot(x='ever_married', kind='count', col='stroke', data=df)
    sns.catplot(x='work_type', kind='count', col='stroke', data=df)
    sns.catplot(x='smoking_status', kind='count', col='stroke', data=df)
    sns.catplot(x='age', kind='count', data=df)


# kind of preprocess
def preprocess(df):
    df = df.drop('id', 1)
    df.ever_married = pd.Series(np.where(df.ever_married == 'Yes', 1, 0), df.index)
    df.gender = pd.Series(np.where(df.gender == 'Male', 1, 0), df.index)

    dff = pd.get_dummies(data=df, columns=['work_type', 'Residence_type','smoking_status'])
    dff = dff.dropna()

    X = dff.drop(['stroke'], axis = 1)
    y = dff.stroke.values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


# the model itself
def adam_model():
    model = Sequential()
    model.add(Dense(12, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def train(X_train, y_train, X_test, y_test):
    model = adam_model()
    history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=10, verbose = 0)

    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    ac=accuracy_score(y_test, y_pred.round())
    print('accuracy of the model: ',ac)
    print('loss: ' + str(history.history['val_loss'][-1]))

    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # Model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Dokładność modelu')
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend(['train', 'test'])
    plt.show()

    # Model Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Funkcja straty')
    plt.ylabel('Strata')
    plt.xlabel('Epoka')
    plt.legend(['train', 'test'])
    plt.show()

def main():
    df = pd.read_csv("train_2v.csv")
    visualization(df)
    #X_train, X_test, y_train, y_test = preprocess(df)
    #train(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
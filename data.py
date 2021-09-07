import pandas as pd
import yfinance as y
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import *
from sklearn.decomposition import *
from pandas_datareader import data as pdr

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std

#Choose arbitrary Tech ETF ticker Apple
apple = y.Ticker("AAPL")

history = apple.history("max")
#print(history)
#print(apple.earnings)

y.pdr_override()
use = pdr.get_data_yahoo(tickers = "AAPL", period = "1d", interval = "1m", auto_adjust = True, prepost = False)
print(use)

use = pd.DataFrame(use)

X = use[["Open", "High", "Low", "Volume"]]
Y = use[["Close"]]

zscore = z_score(use)
print(zscore)

pca = PCA(n_components=3)
pca.fit(zscore)
print(pca.transform(zscore))



#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, shuffle=False)

#X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(100, 'relu', input_shape = (1761,4,1)))
#model.add(tf.keras.layers.Dense(1))
#model.compile("adam", loss = "mse")
#model.fit(X_train, y_train, epochs=100, verbose=1, batch_size = 100, validation_data = (X_test, y_test))

#d = {'predicted': model.predict(X_test), 'actual': y_test}
#print(model.predict(X_test)[1], y_test[1])

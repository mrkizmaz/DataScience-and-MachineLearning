"""
Ersel Kızmaz / ersel.kizmaz@gmail.com

--BTC-USD Borsa Tahmin Uygulaması--
Bu uygulama, linear regresyon modeli ile
bitcoinin dolar kuru bazında degisiminin tahminlemesi yapılmıstır.
"""

# kullanılan kütüphaneler
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# verisetinin cekilmesi ve eksik degerlerden arındırılması
url = "https://mrkizmaz-s3data.s3.eu-west-1.amazonaws.com/DataSets/BTC-USD.csv"
df = pd.read_csv(url)
df = df.dropna()

# verisetinin hazırlanması ve islenmesi
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out) # son satırları eksik olan yeni bir sütun
    X = np.array(df[[forecast_col]]) # sütunu arraye dönüstürme
    X = preprocessing.scale(X) # veriyi isleme
    X_lately = X[-forecast_out:] # tahmin sütunu
    X = X[:-forecast_out] # bagımsız degiskenler
    label.dropna(inplace=True) # eksik degerlerden arındırılır
    y = np.array(label)  # bagımlı degisken
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) # cross validation

    response = [X_train, X_test , Y_train, Y_test, X_lately]
    return response

forecast_col = 'Close' # bagımsız degisken
forecast_out = 5 # tahmin edilen deger sayısı
test_size = 0.2 # verisetinin %20'si test seti

# model kurulumu
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df,forecast_col,forecast_out,test_size)
learner = LinearRegression()
learner.fit(X_train,Y_train)

score = learner.score(X_test,Y_test) # modelin basarı degeri
forecast= learner.predict(X_lately) # tahmin edilen degerler

# sonucları sözlük icine atama islemleri
response = {}
response['test_score'] = score
response['forecast_set'] = forecast

print(response)


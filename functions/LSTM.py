import numpy as np 
import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import yfinance as yf 
import os


def realiza_treinamento(papel):
    
    papel = papel+'.SA'                                                                         ## Adiciona sufixo necessário para chamada do yfinance
    optimizer = Adam(learning_rate=0.001)

    periodo_corte = (datetime.now() - relativedelta(months=10)).strftime('%Y-%m-%d')            ## Seleciona 10 meses como período de treinamento
    data = yf.Ticker(papel).history(start=periodo_corte, end=str(date.today()))
    df_precos = data['Close']

    precos = df_precos.values.reshape(-1, 1)
    scaler = RobustScaler()
    scaler.fit(precos)
    precos_scaled = scaler.transform(precos).flatten()

    X_stg = []
    y_stg = []
    window_size = 10

    for i in range(len(precos_scaled) - window_size):
        
        features = precos_scaled[i : i + window_size]
        target = precos_scaled[i + window_size]
        X_stg.append(features)
        y_stg.append(target)


    X = np.array(X_stg).reshape(-1, window_size, 1) 
    y = np.array(y_stg)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)       ## Separacao do dataset de treinamento e validacao (80/20)

    # Verifica existencia do modelo. Caso já exista, utilizar
    if os.path.exists("./melhor_modelo_ITUB4.SA.h5"):
        model = load_model("melhor_modelo_ITUB4.SA.h5", compile=False)
        model.compile(optimizer=optimizer, loss='mse')
        y_pred = model.predict(X_val)
        y_pred_inv = scaler.inverse_transform(y_pred)

        proximo_fechamnento = round(float((y_pred_inv[-1][0])), 2)
        return proximo_fechamnento
    
    # especifica modelo LSTM
    model = Sequential([
        # Duas camadas de LSTM e dropouts
        LSTM(50, return_sequences=True, input_shape=(10, 1), name="LSTM_1"),
        Dropout(0.2, name="DROP_1"),
        
        LSTM(50, return_sequences=False, name="LSTM_2"),
        Dropout(0.2, name="DROP_2"),
        
        Dense(25, activation='relu', name="DENSE_1"),
        
        Dense(1, name="DENSE_2")
    ], name="MODELO_ACOES")

    model.compile(optimizer=optimizer, loss='mse')
    # model.summary()

    ## especificacao de early stop (fine tune) e model_checkpoint (salva melhor modelo)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('melhor_modelo_'+papel+'.h5', monitor='val_loss', save_best_only=True, verbose=1)
 
    # treinammento
    model.fit(
        X_train,
        y_train,
        epochs=100, 
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, model_checkpoint] 
    )

    # realiza previsao do proximo fechamento
    y_pred = model.predict(X_val)
    y_pred_inv = scaler.inverse_transform(y_pred)

    proximo_fechamnento = round(float((y_pred_inv[-1][0])), 2)
    return proximo_fechamnento

    
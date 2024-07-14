def train():
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import plotly.graph_objects as go

    from datetime import datetime
    import pytz

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)

    generation1 = pd.read_csv("dags/datasets/train/Plant_1_Generation_Data.csv")
    weather1 = pd.read_csv("dags/datasets/train/Plant_1_Weather_Sensor_Data.csv")
    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)


    # In[5]:


    generation1


    # In[6]:


    inverters = list(generation1['SOURCE_KEY'].unique())
    print(f"total number of inverters {len(inverters)}")


    # # Inverter level Anomally detection

    # In[7]:


    inverters[0]


    # In[8]:


    inv_1 = generation1[generation1['SOURCE_KEY']==inverters[0]]
    mask = ((weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"])))
    weather_filtered = weather1.loc[mask]


    # In[9]:


    weather_filtered.shape


    # In[15]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=inv_1["DATE_TIME"], y=inv_1["AC_POWER"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=weather_filtered["DATE_TIME"], y=weather_filtered["IRRADIATION"],
                        mode='lines',
                        name='Irradiation', 
                        yaxis='y2'))

    fig.update_layout(title_text="Irradiation vs AC POWER",
                    yaxis1=dict(title="AC Power in kW",
                                side='left'),
                    yaxis2=dict(title="Irradiation index",
                                side='right',
                                anchor="x",
                                overlaying="y"
                                ))

    fig.write_image(f"dags/outputs/train/{now}_AC_power.png")


    # ### Graph observations
    # We can see that in June 7th and June 14th there are some misproduction areas that could be considered anomalies. Due to the fact that energy production should behave in a linear way to irradiation.

    # In[16]:


    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
    df


    # ### Observations
    # Here we can see how the Isolation Forest Model is behaving. The yellow dots show us the anomalies detected on the test dataset as well as the red squares that show us the anomalies detected on the training dataset. These points do not follow the contour pattern of the graph and we can clearly see that the yellow dots on the far left are the points from June 7th and June 14th.

    # # LSTM Autoencoder approach

    # In[17]:


    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


    # In[18]:


    train_prp = .6
    train = df_.loc[:df_.shape[0]*train_prp]
    test = df_.loc[df_.shape[0]*train_prp:]


    # In[19]:


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")


    # In[20]:


    from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers


    # In[21]:


    def autoencoder_model(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)
        return model


    # In[22]:


    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    model.summary()


    # In[38]:


    epochs = 100
    batch = 10
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch, validation_split=.2, verbose=0).history

    model.save("./models/lstm.h5")


    # In[39]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[x for x in range(len(history['loss']))], y=history['loss'],
                        mode='lines',
                        name='loss'))

    fig.add_trace(go.Scatter(x=[x for x in range(len(history['val_loss']))], y=history['val_loss'],
                        mode='lines',
                        name='validation loss'))

    fig.update_layout(title="Autoencoder error loss over epochs",
                    yaxis=dict(title="Loss"),
                    xaxis=dict(title="Epoch"))

    fig.write_image(f"dags/outputs/train/{now}_error_loss.png")


    # In[40]:


    X_pred = model.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)


    # In[41]:


    scores = pd.DataFrame()
    scores['AC_train'] = train['AC_POWER']
    scores["AC_predicted"] = X_pred["AC_POWER"]
    scores['loss_mae'] = (scores['AC_train']-scores['AC_predicted']).abs()


    # In[42]:


    fig = go.Figure(data=[go.Histogram(x=scores['loss_mae'])])
    fig.update_layout(title="Error distribution", 
                    xaxis=dict(title="Error delta between predicted and real data [AC Power]"),
                    yaxis=dict(title="Data point counts"))
    fig.write_image(f"dags/outputs/train/{now}_error_dist.png")


    # In[43]:


    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)
    X_pred.index = test.index


    # In[44]:


    scores = X_pred
    scores['datetime'] = df_timestamp.loc[1893:]
    scores['real AC'] = test['AC_POWER']
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)


    # In[45]:


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['loss_mae'], 
                            name="Loss"))
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['Threshold'],
                            name="Threshold"))

    fig.update_layout(title="Error Timeseries and Threshold", 
                    xaxis=dict(title="DateTime"),
                    yaxis=dict(title="Loss"))
    fig.write_image(f"dags/outputs/train/{now}_threshold.png")


    # In[46]:


    scores['Anomaly'].value_counts()


    # In[47]:


    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


    # In[51]:


    scores[(scores['Anomaly']==1)&(scores["datetime"].notnull())].to_csv(f"dags/outputs/train/{now}_anomalies.csv")


    # In[48]:


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["real AC"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["anomalies"],
                        name='Anomaly', 
                        mode='markers',
                        marker=dict(color="red",
                                    size=11,
                                    line=dict(color="red",
                                            width=2))))

    fig.update_layout(title_text="Anomalies Detected LSTM Autoencoder")

    fig.write_image(f"dags/outputs/train/{now}_anomaly.png")


    # ## Conclusion
    # 
    # We see that the LSTM Autoencoder approach is a more efficient way to detect anomalies, againts the Isolation Forest approach, perhaps with a larger dataset the Isolation tree could outperform the Autoencoder, having a faster and pretty good model to detect anomalies. 
    # 
    # We can see from the Isolation Forest graph how the model is detecting anomalies, highlighting the datapoints from June 7th and June 14th.
    # 

    # In[ ]:






if __name__ == "__main__":
    train()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go

def inference():
    
    generation1 = pd.read_csv("./datasets/inference/Plant_1_Generation_Data.csv")
    weather1 = pd.read_csv("./datasets/inference/Plant_1_Weather_Sensor_Data.csv")
    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)


    # In[3]:


    generation1


    # In[4]:


    inverters = list(generation1['SOURCE_KEY'].unique())
    print(f"total number of inverters {len(inverters)}")


    # # Inverter level Anomally detection

    # In[5]:


    inverters[0]


    # In[6]:


    inv_1 = generation1[generation1['SOURCE_KEY']==inverters[0]]
    mask = ((weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"])))
    weather_filtered = weather1.loc[mask]


    # In[7]:


    weather_filtered.shape


    # In[8]:


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

    fig.write_image("./outputs/inference/AC_power.png")


    # ### Graph observations
    # We can see that in June 7th and June 14th there are some misproduction areas that could be considered anomalies. Due to the fact that energy production should behave in a linear way to irradiation.

    # In[9]:


    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
    df


    # ### Observations
    # Here we can see how the Isolation Forest Model is behaving. The yellow dots show us the anomalies detected on the test dataset as well as the red squares that show us the anomalies detected on the training dataset. These points do not follow the contour pattern of the graph and we can clearly see that the yellow dots on the far left are the points from June 7th and June 14th.

    # # LSTM Autoencoder approach

    # In[10]:


    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


    # In[17]:


    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    X = scaler.fit_transform(df_.loc[:df_.shape[0]])
    X = X.reshape(X.shape[0], 1, X.shape[1])

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable()
    def mae(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)

    #from tensorflow.keras.models import load_model
    cached_model = tf.keras.models.load_model('./models/lstm.h5', custom_objects={'mae': mae})

    X_pred = cached_model.predict(X)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=df_.columns)
    X_pred.index = df.index

    scores = X_pred.copy()
    scores['datetime'] = df_timestamp
    scores['real AC'] = df['AC_POWER']
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"]>scores["Threshold"], 1, 0)


    # In[18]:


    scores['Anomaly'].value_counts()


    # In[19]:


    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


    # In[20]:


    scores[(scores['Anomaly']==1)&(scores["datetime"].notnull())].to_csv("./outputs/inference/anomalies.csv")


    # In[21]:


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

    fig.write_image("./outputs/inference/anomaly.png")


    # ## Conclusion
    # 
    # We see that the LSTM Autoencoder approach is a more efficient way to detect anomalies, againts the Isolation Forest approach, perhaps with a larger dataset the Isolation tree could outperform the Autoencoder, having a faster and pretty good model to detect anomalies. 
    # 
    # We can see from the Isolation Forest graph how the model is detecting anomalies, highlighting the datapoints from June 7th and June 14th.
    # 

    # In[ ]:






if __name__ == "__main__":
    inference()
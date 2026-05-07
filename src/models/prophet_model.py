from prophet import Prophet


def train_prophet(train, test):

    prophet_train = train[['date', 'sales']]
    prophet_train.columns = ['ds', 'y']

    model = Prophet()

    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(test))

    forecast = model.predict(future)

    preds = forecast.tail(len(test))['yhat']

    return model, preds
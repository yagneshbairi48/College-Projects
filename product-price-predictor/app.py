from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
from datetime import datetime
app = Flask(__name__)

data = pd.read_csv("C:/Users/purna/Downloads/minidata.csv")

data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    product_name = request.form['product_name']
    pdata = data[data['product'] == product_name]
    if pdata.empty:
        return render_template("error.html", message="Product not found")
    else:
        x = pdata[['month', 'competitor_prices']]  # Include 'competitor_prices' as a feature
        y = pdata['price']
        reg = LinearRegression()
        reg.fit(x, y)
        pre = reg.predict(x)
        pdata['predicted_price'] = pre
        min_price = pdata['predicted_price'].min()
        min_index = pdata['predicted_price'].idxmin()
        m = pdata.loc[min_index, 'date'].strftime('%B')

    suggestion = "No idea"

    min_p = pdata['price'].min()
    if min_p < min_price:
        suggestion = "Best time to buy"
    elif min_p > min_price:
        suggestion = "Better to wait"
    else:
        suggestion = "Can be bought"

    image_file = product_name+ '.jpg'
    image_path = os.path.join(app.root_path, 'static', 'images', image_file)
    if not os.path.exists(image_path):
        generate_price_trend_graph(pdata, image_path)

    current_price = pdata['price'].iloc[-1]

    return render_template("result.html", product=product_name, result=int(min_price), month=m, cur_price=current_price, suggestion=suggestion, image=image_file)


def generate_price_trend_graph(data, save_path):

    plt.plot(data['date'], data['price'])

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Trend')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)

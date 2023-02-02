# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, render_template, request

# Load the data
data = pd.read_csv('df.csv')

# Split the data into training and testing sets
X = data[['bookid']]
y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create a Flask app
app = Flask(__name__)


# Define a route to render the form
@app.route('/')
def form():
    return render_template('form.html')


# Define a route to predict the rating
@app.route('/predict', methods=['POST'])
def predict():
    # Get the book_id from the form
    book_id = request.form['bookid']

    # Use the model to make a prediction
    prediction = model.predict([[book_id]])

    # Return the prediction as a response
    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')

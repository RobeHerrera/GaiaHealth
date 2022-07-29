from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def ai_model():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        first_name = request.form.get("name")
        # getting input with name = lname in HTML form
        email = request.form.get("email")
        return "Your name is "+ first_name + " " + email
    return "recived: {}".format(request.form)


# TODO Robert: delete this, only for demostration proposess
@app.route("/<celsius>")
def fahrenheit_from(celsius):
    """Convert Celsius to Fahrenheit degrees."""
    try:
        fahrenheit = float(celsius) * 9 / 5 + 32
        fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
        return str(fahrenheit)
    except ValueError:
        return "invalid input"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
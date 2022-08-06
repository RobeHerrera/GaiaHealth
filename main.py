from flask import Flask
from flask import request
from flask import render_template
from model import breast_model as model

app = Flask(__name__, template_folder='./templates')

# Point of improvement
# 1- Have multiple data sets to change in the dropdown menu

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/post', methods=['GET', 'POST'])
def ai_model():
    if request.method == "POST":
        # TODO Rober start here tomorrow
        # 1 - Create a list
        # 2 - List to Pandas DataFrame
        # 3 - Send the DF to model
        # 4 - Print Prediction
        # 5 - Display Prediction in the HTML

        # getting input with name = fname in HTML form
        first_name = request.form.get("radious_mean")
        # getting input with name = lname in HTML form
        email = request.form.get("texture_mean")
        return "Your name is " + str(first_name) + " " + str(email)
    return "recived: {}".format(request.form)


if __name__ == "__main__":
    model.run()
    app.run(host="127.0.0.1", port=8080, debug=True)

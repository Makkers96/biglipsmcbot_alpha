from flask import Flask, render_template, request, session
from main import run_llm
import os

app = Flask(__name__)
# app.secret_key = os.getenv('secret_key')
app.secret_key = 'temporary_test_key'


@app.route("/", methods=["GET", "POST"])
def homepage():
    return "This is the homepage."


@app.route("/bot", methods=["GET", "POST"])
def bot():

    if 'messages' not in session:
        session['messages'] = [{'sender': 'Big Lips McBot', 'message': 'Hello! How can I help you?'}]

    if request.method == 'POST':
        if 'user_input' in request.form:


            # add user input to a variable
            session['user_input'] = request.form.get('user_input')
            print(f"ERROR TEST: Session User Input: {session['user_input']}")

            # run it through the llm
            session['llm_response'] = run_llm(session['user_input'])
            print(f"ERROR TEST: Session llm response: {session['llm_response']}")

            # append messages to the session
            session['messages'].append({'sender': 'user', 'message': session['user_input']})
            session['messages'].append({'sender': 'Big Lips McBot', 'message': session['llm_response']})


    return render_template("bot.html",
                           messages=session['messages'],
                           )



if __name__ == "__main__":
    app.run()

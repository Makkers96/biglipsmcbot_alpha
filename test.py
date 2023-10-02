from flask import Flask, render_template, request, session
import os

app = Flask(__name__)
app.secret_key = os.getenv('secret_key')
print(app.secret_key)
from flask import Flask, render_template, request
from matcher import match_jobs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        profile = request.form['profile']
        results = match_jobs(profile)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
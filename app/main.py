import sampleRecommendation
from flask import Flask, request, render_template, url_for, redirect, jsonify
from flask_cors import CORS

import os
from datetime import datetime

os.chdir(os.path.abspath(os.path.dirname(__file__)))

TEMPLATE_DIR = os.path.abspath('./www')
STATIC_DIR = os.path.abspath('./www')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)
print("Server App Running")

@app.before_first_request
def run_first_model():
    sampleRecommendation.getData()
    return None


@app.route('/', methods=['GET'])
def main_page():
    if request.method == 'GET':
        return render_template('index.html')
    return jsonify({"error": "Only GET request method"})


@app.route('/api/sample/predict', methods=['GET'])
def predict_api():
    if request.method == 'GET':
        id = request.args.get("id", default=1, type = int)
    else:
        return jsonify({"error": "Only GET request method"})

    return jsonify({
        "sourceId": id,
        "sourceData": sampleRecommendation.getById(id),
        "predictedList": sampleRecommendation.predictRecommends(id, 10)
    })


@app.route('/api/sample/list', methods=['GET'])
def get_list():
    if request.method == 'GET':
        start = request.args.get("start", default=0, type = int)
        end = request.args.get("end", default=20, type = int)
    else:
        return jsonify({"error": "Only GET request method"})

    return jsonify({
        "start": start,
        "end": end,
        "list": sampleRecommendation.getIds(start, end)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6789)

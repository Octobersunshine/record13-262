import io
import math
import os

import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_correlations(df):
    numeric_df = df.select_dtypes(include=['number'])
    columns = numeric_df.columns.tolist()

    def sanitize(matrix):
        return [
            [None if math.isnan(v) else v for v in row]
            for row in matrix.values.tolist()
        ]

    def matrix_to_dict(df_matrix):
        return {
            'columns': columns,
            'data': sanitize(df_matrix)
        }

    return {
        'columns': columns,
        'pearson': matrix_to_dict(numeric_df.corr(method='pearson', min_periods=1)),
        'spearman': matrix_to_dict(numeric_df.corr(method='spearman', min_periods=1))
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/correlation', methods=['POST'])
def correlation():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400

    if df.select_dtypes(include=['number']).shape[1] < 2:
        return jsonify({'error': 'CSV must contain at least 2 numeric columns'}), 400

    try:
        result = compute_correlations(df)
    except Exception as e:
        return jsonify({'error': f'Failed to compute correlations: {str(e)}'}), 500

    return jsonify({
        'rows': len(df),
        'numeric_columns': result['columns'],
        'pearson': result['pearson'],
        'spearman': result['spearman']
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

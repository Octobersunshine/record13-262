import io
import math
import os

import pandas as pd
from flask import Flask, request, jsonify, render_template

HEATMAP_STOPS = [
    (-1.0, (49, 54, 149)),
    (-0.5, (69, 117, 180)),
    (0.0, (247, 247, 247)),
    (0.5, (215, 48, 39)),
    (1.0, (165, 0, 38)),
]


def value_to_color(val):
    if val is None:
        return (220, 220, 220)
    v = max(-1.0, min(1.0, val))
    for i in range(len(HEATMAP_STOPS) - 1):
        v0, c0 = HEATMAP_STOPS[i]
        v1, c1 = HEATMAP_STOPS[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)
    return HEATMAP_STOPS[-1][1]


def build_heatmap(sanitized_data):
    colors = []
    for row in sanitized_data:
        color_row = []
        for v in row:
            r, g, b = value_to_color(v)
            color_row.append(f'rgb({r},{g},{b})')
        colors.append(color_row)
    legend_stops = []
    for val, (r, g, b) in HEATMAP_STOPS:
        legend_stops.append({
            'value': val,
            'color': f'rgb({r},{g},{b})'
        })
    return {
        'colors': colors,
        'legend': {
            'min': -1.0,
            'max': 1.0,
            'stops': legend_stops
        }
    }


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
        s = sanitize(df_matrix)
        return {
            'columns': columns,
            'data': s,
            'heatmap': build_heatmap(s)
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

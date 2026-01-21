
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    image_data = data.get('image')

    # Here you would send the image_data to your server for processing
    # For now, we'll just return a dummy response
    print(f"Received image data (first 50 chars): {image_data[:50]}...")
    return jsonify({'status': 'success', 'message': 'Image successfully received by server!', 'result': 'Dummy result for now'})

if __name__ == '__main__':
    app.run(debug=True)

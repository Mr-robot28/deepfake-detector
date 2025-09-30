import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, render_template
from backend.predict import predict_image

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            save_path = os.path.join("data", "raw", file.filename)
            os.makedirs("data/raw", exist_ok=True)
            file.save(save_path)
            result = predict_image(save_path)
            return render_template("index.html", result=result, fname=file.filename)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False)

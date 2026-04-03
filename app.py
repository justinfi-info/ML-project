from flask import Flask, render_template, request

from pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    data = CustomData(
        gender=request.form.get("gender", ""),
        race_ethnicity=request.form.get("race_ethnicity", ""),
        parental_level_of_education=request.form.get("parental_level_of_education", ""),
        lunch=request.form.get("lunch", ""),
        test_preparation_course=request.form.get("test_preparation_course", ""),
        reading_score=float(request.form.get("reading_score", 0)),
        writing_score=float(request.form.get("writing_score", 0)),
    )

    pred_df = data.to_dataframe()
    result = PredictPipeline().predict(pred_df)[0]
    return render_template("home.html", results=float(result))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

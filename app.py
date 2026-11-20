    </section>

    <aside class="side">
      <div class="dark-card">
        <div class="dark-kicker">LOANWISE INTELLIGENCE</div>
        <h3>A smarter way to explore lending outcomes.</h3>
        <p>Submit applicant information and receive a prediction from the existing Decision Tree model. No model training happens during prediction.</p>
      </div>
      <div class="info">
        <h3>How it works</h3>
        <p>Three simple steps to a model prediction:</p>
        <ul><li>Enter all 13 applicant features.</li><li>Submit the information securely to Flask.</li><li>The saved model returns its predicted class.</li></ul>
      </div>
      <div class="notice"><strong>Important:</strong> This is an educational portfolio demonstration, not a real lending decision or financial advice. Model predictions and probabilities are estimates, not guarantees. Do not use this demo to make actual credit decisions.</div>
    </aside>
  </main>
  <footer>LoanWise · Machine Learning Portfolio Project · Built with Flask &amp; Scikit-learn</footer>

  <div class="loader" id="loader"><div class="loader-box"><div class="spinner"></div><strong>Analysing applicant information…</strong></div></div>
  <script>
    const form = document.getElementById('predictionForm');
    const loader = document.getElementById('loader');
    form.addEventListener('submit', () => {
      if (form.reportValidity()) loader.classList.add('show');
    });
    document.getElementById('resetBtn').addEventListener('click', () => {
      const result = document.getElementById('predictionResult');
      if (result) result.remove();
      if (loader) loader.classList.remove('show');
    });
    const result = document.getElementById('predictionResult');
    if (result) result.scrollIntoView({behavior:'smooth', block:'center'});
  </script>
</body>
</html>
"""


def option(value, label, selected):
    is_selected = "selected" if value == selected else ""
    # Values and labels come only from the fixed option lists above.
    return f'<option value="{value}" {is_selected}>{label}</option>'


@app.route("/", methods=["GET", "POST"])
def home():
    values = request.form.to_dict() if request.method == "POST" else {}
    prediction = None
    prediction_label = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            encoded_values = {}

            for field in NUMERIC_FIELDS:
                raw_value = request.form.get(field, "").strip()
                if not raw_value:
                    raise ValueError(f"Please enter a value for {field}.")
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError(f"{field} must be a valid finite number.")
                encoded_values[field] = value

            for field, mapping in ENCODINGS.items():
                raw_value = request.form.get(field, "")
                if raw_value not in mapping:
                    raise ValueError(f"Please select a valid option for {field}.")
                encoded_values[field] = mapping[raw_value]

            # Preserve the exact feature order used when the model was trained.
            row = [[encoded_values[feature] for feature in FEATURES]]

            # DataFrame retains the training feature names as well as their order.
            input_data = pd.DataFrame(row, columns=FEATURES)

            predicted_class = model.predict(input_data)[0]
            prediction = int(predicted_class)

            if prediction == 1:
                prediction_label = "Approved (1)"
            elif prediction == 0:
                prediction_label = "Rejected (0)"
            else:
                prediction_label = str(prediction)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                class_index = list(model.classes_).index(predicted_class)
                confidence = round(float(probabilities[class_index]) * 100, 2)

        except (TypeError, ValueError) as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "The model could not process this input. Check that the "
                "category encodings and input values match the training data."
            )

    return render_template_string(
        PAGE,
        values=values,
        prediction=prediction,
        prediction_label=prediction_label,
        confidence=confidence,
        error=error,
        option=option,
    )


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "features": FEATURES,
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

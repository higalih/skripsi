import os

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from kaggle.api.kaggle_api_extended import KaggleApi
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Daily nutrition standard (FDA/WHO)
DAILY_NUTRITION = {
    "calories": 2000,  # kcal
    "fat": 70,  # g
    "carbohydrates": 300,  # g
    "protein": 50,  # g
    "cholesterol": 300,  # mg
    "sodium": 2300,  # mg
    "fiber": 28,  # g
}

app = Flask(__name__, template_folder="views")
app.secret_key = "your_secret_key"  # Needed for session

# ===== KAGGLE IMAGE PROXY SETUP ===== #

# Set config directory explicitly (where kaggle.json is located)
os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))

# Initialize Kaggle API
kaggle_api = None
try:
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    print("Kaggle API authenticated successfully")
except Exception as e:
    print(f"Kaggle API error: {str(e)}")
    kaggle_api = None


@app.route("/kaggle_image/<int:recipe_id>")
def serve_image(recipe_id):
    """
    Proxy route for Kaggle-hosted images with local caching in /tmp.
    """
    temp_path = f"/tmp/{recipe_id}.jpg"
    if not os.path.exists(temp_path):
        if kaggle_api:
            try:
                kaggle_api.dataset_download_file(
                    "elisaxxygao/foodrecsysv1",
                    f"raw-data-images/raw-data-images/{recipe_id}.jpg",
                    path="/tmp",
                    quiet=True,
                    force=True,
                )
                # The Kaggle API downloads as a zip if not a single file, so check and extract if needed
                zip_path = temp_path + ".zip"
                if os.path.exists(zip_path):
                    import zipfile

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall("/tmp")
                    os.remove(zip_path)
            except Exception as e:
                print(f"Kaggle API image fetch failed: {str(e)}")
                return "Image unavailable", 404
        else:
            return "Image unavailable", 404
    try:
        return send_file(temp_path, mimetype="image/jpeg")
    except Exception as e:
        print(f"Image send failed: {str(e)}")
        return "Image unavailable", 404


# Load data
data = pd.read_csv("recipes_kaggle_images.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer(max_features=500)  # Limit features for memory
X_ingredients = vectorizer.fit_transform(data["ingredients_list"])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(
    data[
        [
            "calories",
            "fat",
            "carbohydrates",
            "protein",
            "cholesterol",
            "sodium",
            "fiber",
        ]
    ]
)

# Combine Features (keep everything sparse)
X_numerical_sparse = csr_matrix(X_numerical)
X_combined = hstack([X_numerical_sparse, X_ingredients])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(X_combined)


def recommend_recipes(input_features):
    input_features_scaled = scaler.transform([input_features[:7]])
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    input_numerical_sparse = csr_matrix(input_features_scaled)
    input_combined = hstack([input_numerical_sparse, input_ingredients_transformed])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[
        ["recipe_name", "ingredients_list", "image_url", "recipe_id"]
    ].head(5)


def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


@app.route("/", methods=["GET", "POST"])
def index():
    if "meals" not in session:
        session["meals"] = []

    error = None
    recommendations = []
    if request.method == "POST":
        try:
            # Validate and parse inputs
            calories = float(request.form.get("calories", 0))
            fat = float(request.form.get("fat", 0))
            carbohydrates = float(request.form.get("carbohydrates", 0))
            protein = float(request.form.get("protein", 0))
            cholesterol = float(request.form.get("cholesterol", 0))
            sodium = float(request.form.get("sodium", 0))
            fiber = float(request.form.get("fiber", 0))
            ingredients = request.form.get("ingredients", "").strip()

            if not ingredients:
                raise ValueError("Ingredients cannot be empty.")

            meal = {
                "calories": calories,
                "fat": fat,
                "carbohydrates": carbohydrates,
                "protein": protein,
                "cholesterol": cholesterol,
                "sodium": sodium,
                "fiber": fiber,
                "ingredients": ingredients,
            }
            meals = session["meals"]
            if len(meals) < 3:
                meals.append(meal)
                session["meals"] = meals
            else:
                error = "You have already logged 3 meals for today. Please reset to start a new day."

            input_features = [
                calories,
                fat,
                carbohydrates,
                protein,
                cholesterol,
                sodium,
                fiber,
                ingredients,
            ]

            recommendations = recommend_recipes(input_features).to_dict(
                orient="records"
            )
        except ValueError as e:
            error = str(e)
        except Exception:
            error = "An unexpected error occurred. Please try again."

    # Nutrition summary calculation
    totals = {key: 0 for key in DAILY_NUTRITION}
    for meal in session["meals"]:
        for key in totals:
            totals[key] += float(meal.get(key, 0))
    remaining = {}
    warnings = []
    for key in DAILY_NUTRITION:
        remain = DAILY_NUTRITION[key] - totals[key]
        remaining[key] = remain
        if remain < 0:
            warnings.append(f"{key.capitalize()} exceeded by {abs(remain):.2f}")
        elif remain < DAILY_NUTRITION[key] * 0.1:
            warnings.append(f"{key.capitalize()} almost reached the limit.")

    return render_template(
        "index.html",
        recommendations=recommendations,
        truncate=truncate,
        totals=totals,
        remaining=remaining,
        warnings=warnings,
        error=error,
        DAILY_NUTRITION=DAILY_NUTRITION,
    )


@app.route("/reset", methods=["POST"])
def reset():
    session["meals"] = []
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

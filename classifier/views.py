import os
import django
import joblib
import pandas as pd
from django.shortcuts import render,redirect
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Prediction


# Paths


def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect('signup')

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        return redirect('login')

    return render(request, 'signup.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid credentials")
            return redirect('login')

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')
    
@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'predictions': predictions})

BASE_DIR = settings.BASE_DIR
APP_DIR = os.path.join(BASE_DIR, "classifier")


DATASET_PATH = os.path.join(
    APP_DIR,
    "mecfs_depression_dataset_80000_rows-1.csv"
)
MODEL_PATH = os.path.join(APP_DIR, "rf_model.pkl")
ENCODER_PATH = os.path.join(APP_DIR, "label_encoder.pkl")




# Model Training (only if model not saved)

def train_and_save_model():
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna()

    X = data.drop("label", axis=1)
    y = data["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Classes Found:", encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model trained successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classes Learned: {encoder.classes_}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("Model and encoder saved successfully.")




# Load Model


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        train_and_save_model()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


# Suggestion Engine


def generate_suggestions(prediction_label):
    suggestions_map = {
        "Depression": [
            "Maintain a regular sleep schedule",
            "Engage in moderate physical activity",
            "Practice mindfulness or breathing exercises",
            "Stay socially connected",
            "Consult a mental health professional if symptoms persist"
        ],
        "ME/CFS": [
            "Avoid overexertion and manage energy levels",
            "Follow a balanced nutrition plan",
            "Prioritize quality sleep hygiene",
            "Monitor symptoms consistently",
            "Consult a specialist for tailored management"
        ],
         "Normal": [
        "You appear to be healthy. Keep maintaining your lifestyle.",
        "Avoid overthinking about symptoms.",
        "Continue regular exercise and balanced diet.",
        "Maintain good sleep habits.",
        "Stay positive and take care of your mental wellbeing."
    ]
    }

    return suggestions_map.get(prediction_label, [])



# Main View


@login_required
def home(request):
    model, encoder = load_model()

    context = {}

    if request.method == "POST":
        try:
            input_features = [
                int(request.POST.get("fatigue_score")),
                int(request.POST.get("sleep_disturbance")),
                int(request.POST.get("muscle_pain")),
                int(request.POST.get("joint_pain")),
                int(request.POST.get("headache_frequency")),
                int(request.POST.get("memory_problems")),
                int(request.POST.get("concentration_issues")),
                int(request.POST.get("mood_score")),
                int(request.POST.get("anxiety_level")),
                int(request.POST.get("post_exertional_malaise")),
            ]

            prediction = model.predict([input_features])[0]

            probabilities = model.predict_proba([input_features])[0]
            probability = max(probabilities)

            label = encoder.inverse_transform([prediction])[0]

            if probability < 0.55:
                label = "Normal"

            Prediction.objects.create(
                user=request.user,
                input_data=str(input_features),
                result=label
            )

            context = {
                "prediction": label,
                "probability": round(probability * 100, 2),
                "suggestions": generate_suggestions(label)
            }

        except Exception as e:
            context["error"] = str(e)

    return render(request, "home.html", context)

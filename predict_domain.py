import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('tech_domain_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['features']
domain_mapping = model_data['domain_mapping']

def predict_domain(tags, score=0, answer_count=0, year=2023):
    """Predict technology domain for a question"""
    
    # Create feature vector
    features = {
        'Score': score,
        'AnswerCount': answer_count,
        'TagCount': len(tags),
        'ScorePerAnswer': score / (answer_count + 1) if answer_count > 0 else score,
        'Year': year
    }
    
    # Add tag features
    for col in feature_cols:
        if col.startswith('has_'):
            tag_name = col[4:]  # Remove 'has_' prefix
            features[col] = 1 if tag_name in tags else 0
    
    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[feature_cols]
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return prediction, dict(zip(model.classes_, probability))

# Example predictions
examples = [
    (['python', 'pandas', 'machine-learning'], 15, 3),
    (['javascript', 'html', 'css'], 8, 2),
    (['java', 'spring', 'hibernate'], 12, 4),
    (['android', 'kotlin'], 5, 1),
    (['sql', 'mysql', 'database'], 10, 2)
]

print("Technology Domain Predictions:")
print("=" * 50)

for tags, score, answers in examples:
    domain, probabilities = predict_domain(tags, score, answers)
    print(f"\nTags: {tags}")
    print(f"Predicted Domain: {domain}")
    print(f"Confidence: {probabilities[domain]:.3f}")
    print(f"All probabilities: {dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))}")
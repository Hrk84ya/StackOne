import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import pickle

# Load and sample data (using subset for faster processing)
print("Loading data...")
questions = pd.read_csv('data/questions.csv', nrows=100000)
tags = pd.read_csv('data/question_tags.csv')

# Filter tags for sampled questions
tags = tags[tags['Id'].isin(questions['Id'])]

# Technology domain mapping
domain_mapping = {
    'Web Development': ['html', 'css', 'javascript', 'php', 'jquery', 'ajax', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django', 'flask', 'asp.net', 'laravel'],
    'Mobile Development': ['android', 'ios', 'swift', 'kotlin', 'react-native', 'flutter', 'xamarin', 'cordova', 'ionic'],
    'Data Science': ['python', 'r', 'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'tensorflow', 'pytorch', 'machine-learning', 'data-analysis'],
    'Backend/Systems': ['java', 'c++', 'c#', 'go', 'rust', 'scala', 'spring', 'hibernate', 'microservices', 'api'],
    'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'database', 'nosql'],
    'DevOps/Cloud': ['docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'git', 'linux', 'bash', 'deployment']
}

def get_domain(question_tags):
    """Assign domain based on tags"""
    for domain, domain_tags in domain_mapping.items():
        if any(tag in domain_tags for tag in question_tags):
            return domain
    return 'Other'

# Group tags by question
question_tags = tags.groupby('Id')['Tag'].apply(list).reset_index()
data = questions.merge(question_tags, on='Id', how='inner')

# Create target variable
data['Domain'] = data['Tag'].apply(get_domain)

# Remove 'Other' category and questions with missing data
data = data[data['Domain'] != 'Other']
data = data.dropna(subset=['Score', 'AnswerCount'])

print(f"Domain distribution:\n{data['Domain'].value_counts()}")

# Feature engineering
data['TagCount'] = data['Tag'].apply(len)
data['ScorePerAnswer'] = np.where(data['AnswerCount'] > 0, data['Score'] / data['AnswerCount'], data['Score'])
data['Year'] = pd.to_datetime(data['CreationDate']).dt.year

# Fill missing values and handle infinities
data['Score'] = data['Score'].fillna(0)
data['AnswerCount'] = data['AnswerCount'].fillna(0)
data['ScorePerAnswer'] = data['ScorePerAnswer'].replace([np.inf, -np.inf], 0)

print(f"Final dataset size: {len(data)}")

# Create tag frequency features
all_tags = [tag for tags_list in data['Tag'] for tag in tags_list]
top_tags = [tag for tag, count in Counter(all_tags).most_common(50)]

for tag in top_tags:
    data[f'has_{tag}'] = data['Tag'].apply(lambda x: 1 if tag in x else 0)

# Select features
feature_cols = ['Score', 'AnswerCount', 'TagCount', 'ScorePerAnswer', 'Year'] + [f'has_{tag}' for tag in top_tags]
X = data[feature_cols].fillna(0)
y = data['Domain']

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save model
with open('tech_domain_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': feature_cols, 'domain_mapping': domain_mapping}, f)

print("\nModel saved as 'tech_domain_model.pkl'")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("1. Loading ASL Dataset...")
try:
    # Load the data you collected
    df = pd.read_csv('asl_dataset.csv')
except FileNotFoundError:
    print("❌ Error: 'asl_dataset.csv' not found. Did you run the collection script?")
    exit()

# 2. Prepare the Data
# 'y' is what we want to predict (the letter label)
# 'X' is the data we use to predict it (the 63 coordinates)
y = df['label']
X = df.drop('label', axis=1)

# Split the data: 80% for training the AI, 20% for testing its knowledge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training the AI Model (Random Forest)...")
# Initialize the Random Forest algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on your hand coordinates!
model.fit(X_train, y_train)

# 3. Test the Accuracy
print("3. Testing Model Accuracy...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Training Complete! AI Accuracy: {accuracy * 100:.2f}%")

# 4. Save the "Brain"
print("4. Saving the model to 'asl_model.pkl'...")
with open('asl_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Done! You are ready for real-time translation.")

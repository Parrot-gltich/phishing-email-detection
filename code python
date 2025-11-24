import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Création du dataset
# ============================

emails_data = {
    'text': [
        "Congratulations! You have won $1,000,000. Click here to claim your prize now!",
        "Your account has been compromised. Verify your identity immediately by clicking this link.",
        "Hi team, please find attached the quarterly report for review.",
        "Meeting scheduled for tomorrow at 10 AM in conference room B.",
        "URGENT: Your bank account will be closed. Update your information now!",
        "Dear customer, your package is ready for delivery. Track your shipment here.",
        "Hello, I hope this email finds you well. Let's schedule a call next week.",
        "You have inherited $5 million from a distant relative. Contact us immediately!",
        "Reminder: Project deadline is next Friday. Please submit your work on time.",
        "Your password will expire today. Reset it now to avoid account suspension!",
        "Thank you for your purchase. Your order #12345 will arrive in 3-5 business days.",
        "WINNER ALERT! You are the lucky winner of our lottery. Claim your prize!",
        "Can you please review the attached document and provide your feedback?",
        "Your Netflix subscription has expired. Update payment information to continue.",
        "Team lunch tomorrow at 12:30 PM. Please confirm your attendance.",
        "FINAL NOTICE: Your account will be deleted unless you verify your email now!",
        "Please find the meeting notes from yesterday's discussion attached.",
        "You have received a secure message. Click to view your encrypted document.",
        "Quarterly performance review scheduled for next Monday at 2 PM.",
        "Act now! Limited time offer. Get 90% discount on all products today!"
    ],
    'label': [
        1, 1, 0, 0, 1,
        0, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(emails_data)

print("Aperçu du dataset :")
print(df.head())

print("\nDistribution des classes :")
print(df['label'].value_counts())

# ============================
# 2. Prétraitement & Vectorisation
# ============================

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Taille train : {len(X_train)}")
print(f"Taille test : {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nDimensions TF-IDF : ", X_train_tfidf.shape)

# ============================
# 3. Entraînement du modèle
# ============================

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("\n✓ Modèle entraîné avec succès !")

# ============================
# 4. Évaluation du modèle
# ============================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Légitime', 'Phishing']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Légitime', 'Phishing'],
            yticklabels=['Légitime', 'Phishing'])
plt.title("Matrice de confusion - Détection de phishing")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.tight_layout()
plt.savefig("confusion_matrix_phishing.png", dpi=300)
plt.show()

print("\n✓ Matrice de confusion enregistrée sous 'confusion_matrix_phishing.png'")

# ============================
# 5. Test sur nouveaux emails
# ============================

nouveaux_emails = [
    "Your Amazon order has been shipped. Track your package here.",
    "URGENT! Your account has been locked. Verify now to unlock!",
    "Meeting rescheduled to Thursday at 3 PM. Please update your calendar."
]

nouveaux_tfidf = vectorizer.transform(nouveaux_emails)
preds = model.predict(nouveaux_tfidf)

for email, p in zip(nouveaux_emails, preds):
    label = "⚠️ PHISHING" if p == 1 else "✓ LÉGITIME"
    print("\n", label)
    print("Email :", email[:100], "...")

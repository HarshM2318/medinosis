from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Initialize the Flask application
app = Flask(__name__)

# Step 1: Load the symptom CSV files (Training and Testing data)
df_train = pd.read_csv('Prototype.csv')
df_test = pd.read_csv('Prototype1.csv')

# Step 2: Generate the disease mapping from the 'prognosis' column
unique_diseases = df_train['prognosis'].unique()
disease_dict = {disease: idx for idx, disease in enumerate(unique_diseases)}
reverse_disease_dict = {v: k for k, v in disease_dict.items()}  # To convert ID back to disease name

# Replace disease names with numerical values in both train and test datasets
df_train['prognosis'] = df_train['prognosis'].map(disease_dict)
df_test['prognosis'] = df_test['prognosis'].map(disease_dict)

# Prepare the training and testing data
#l1 = ['itching', 'skin_rash', 'continuous_sneezing', 'fatigue', 'high_fever']  # List of symptoms

l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
      'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 
      'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
      'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
      'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 
      'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 
      'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
      'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
      'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 
      'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 
      'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
      'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 
      'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
      'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremities', 
      'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 
      'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
      'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
      'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
      'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
      'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 
      'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 
      'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
      'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 
      'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 
      'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
      'red_sore_around_nose', 'yellow_crust_ooze']

# Filter l1 to only include symptoms that exist in df_train
l1 = [symptom for symptom in l1 if symptom in df_train.columns]


X_train = df_train[l1]
y_train = df_train['prognosis']

X_test = df_test[l1]
y_test = df_test['prognosis']

# Helper function to map symptoms to binary input
def get_symptom_input(symptoms, all_symptoms):
    symptom_vector = [0] * len(all_symptoms)
    for symptom in symptoms:
        if symptom in all_symptoms:
            symptom_vector[all_symptoms.index(symptom)] = 1
    return [symptom_vector]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html', symptoms=l1)

# Route for making predictions using all three models
@app.route('/predict', methods=['POST'])
def predict():
    psymptoms = [request.form['symptom1'], request.form['symptom2'], request.form['symptom3'], request.form['symptom4'], request.form['symptom5']]

    # Prepare input for prediction
    input_test = get_symptom_input(psymptoms, l1)

    # Train and predict using Decision Tree
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(X_train, y_train)
    prediction_tree = clf_tree.predict(input_test)[0]
    disease_tree = reverse_disease_dict.get(prediction_tree, "Disease Not Found")

    # Train and predict using Random Forest
    clf_rf = RandomForestClassifier()
    clf_rf = clf_rf.fit(X_train, y_train)
    prediction_rf = clf_rf.predict(input_test)[0]
    disease_rf = reverse_disease_dict.get(prediction_rf, "Disease Not Found")

    # Train and predict using Naive Bayes
    clf_nb = GaussianNB()
    clf_nb = clf_nb.fit(X_train, y_train)
    prediction_nb = clf_nb.predict(input_test)[0]
    disease_nb = reverse_disease_dict.get(prediction_nb, "Disease Not Found")

    # Render the result page with all three predictions
    return render_template('result.html', 
                           prediction_tree=disease_tree,
                           prediction_rf=disease_rf,
                           prediction_nb=disease_nb)

if __name__ == "__main__":
    app.run(debug=True)

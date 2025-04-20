import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt

# Load models
reg_model = joblib.load('student_performance_reg_model.pkl')
clf_model = joblib.load('student_performance_clf_model.pkl')

# Prediction function
def predict_student_performance(input_data, reg_model, clf_model):
    input_df = pd.DataFrame([input_data])
    reg_pred = reg_model.predict(input_df)[0]
    clf_pred = clf_model.predict(input_df)[0]
    clf_proba = clf_model.predict_proba(input_df)[0][1]
    return reg_pred, clf_pred, clf_proba

# Streamlit app
st.title("Student Performance Prediction Dashboard")

st.header("Input Student Data")
math_score = st.slider("Math Score", 0, 100, 50)
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_education = st.selectbox("Parental Education", ["some high school", "high school", "some college", 
                                                       "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation", ["none", "completed"])

# Prepare input data
input_data = {
    'math score': math_score,
    'reading score': reading_score,
    'writing score': writing_score,
    'gender_encoded': 70.0 if gender == "male" else 65.0,  # Approximate from target encoding
    'race/ethnicity_encoded': {"group A": 60.0, "group B": 65.0, "group C": 67.0, "group D": 68.0, "group E": 70.0}[race],
    'parental level of education_encoded': {"some high school": 60.0, "high school": 62.0, "some college": 65.0, 
                                           "associate's degree": 67.0, "bachelor's degree": 70.0, "master's degree": 72.0}[parent_education],
    'lunch_encoded': 70.0 if lunch == "standard" else 60.0,
    'test preparation course_encoded': 70.0 if test_prep == "completed" else 65.0,
    'gender_test_prep_encoded': 70.0 if f"{gender}_{test_prep}" in ["male_completed", "female_completed"] else 65.0
}

# Predict
reg_pred, clf_pred, clf_proba = predict_student_performance(input_data, reg_model, clf_model)
st.header("Prediction Results")
st.write(f"Predicted Average Score: {reg_pred:.2f}")
st.write(f"Pass/Fail: {'Pass' if clf_pred else 'Fail'}")
st.write(f"Pass Probability: {clf_proba:.2f}")

# SHAP Explanation
st.header("Feature Importance (SHAP)")
X_input = pd.DataFrame([input_data])
X_input_transformed = reg_model.named_steps['preprocessor'].transform(X_input)
explainer = shap.KernelExplainer(reg_model.named_steps['stacking'].predict, reg_model.named_steps['preprocessor'].transform(X_input))
shap_values = explainer.shap_values(X_input_transformed, nsamples=50)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_input, feature_names=X_input.columns, plot_type="bar", show=False)
plt.savefig('shap_input.png')
st.image('shap_input.png')

# Plotly Visualization
st.header("Interactive Data Exploration")
df = pd.read_csv('StudentsPerformance.csv')
df['avg_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
fig = px.scatter(df, x='math score', y='avg_score', color='test preparation course', 
                 size='reading score', hover_data=['gender', 'race/ethnicity'],
                 title='Math Score vs. Average Score by Test Preparation')
st.plotly_chart(fig)
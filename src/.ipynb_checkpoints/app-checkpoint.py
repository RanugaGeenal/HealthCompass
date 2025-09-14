import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="HealthCompass - Insurance Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .bmi-normal { color: #28a745; }
    .bmi-overweight { color: #ffc107; }
    .bmi-obese { color: #dc3545; }
    .bmi-underweight { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("E:/HealthCompass/notebooks/decision_tree_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is in the correct location.")
        return None

model = load_model()

def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "bmi-underweight"
    elif bmi < 25:
        return "Normal weight", "bmi-normal"
    elif bmi < 30:
        return "Overweight", "bmi-overweight"
    else:
        return "Obese", "bmi-obese"

def bmi_calculator():
    st.markdown('<h2 class="sub-header">🧮 BMI Calculator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Your Measurements")
        height_unit = st.selectbox("Height Unit", ["cm", "ft/in"])
        
        if height_unit == "cm":
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            height_m = height_cm / 100
        else:
            feet = st.number_input("Feet", min_value=3, max_value=8, value=5)
            inches = st.number_input("Inches", min_value=0, max_value=11, value=8)
            height_m = (feet * 12 + inches) * 0.0254
        
        weight_unit = st.selectbox("Weight Unit", ["kg", "lbs"])
        
        if weight_unit == "kg":
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
        else:
            weight_lbs = st.number_input("Weight (lbs)", min_value=66.0, max_value=660.0, value=154.0)
            weight_kg = weight_lbs * 0.453592
        
        if st.button("Calculate BMI", type="primary"):
            bmi = weight_kg / (height_m ** 2)
            category, css_class = calculate_bmi_category(bmi)
            
            with col2:
                st.subheader("Results")
                st.metric("BMI", f"{bmi:.1f}")
                st.markdown(f'<p class="{css_class}"><strong>Category: {category}</strong></p>', unsafe_allow_html=True)
                
                # BMI gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = bmi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "BMI"},
                    delta = {'reference': 22.5},
                    gauge = {
                        'axis': {'range': [None, 40]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 18.5], 'color': "lightgray"},
                            {'range': [18.5, 25], 'color': "lightgreen"},
                            {'range': [25, 30], 'color': "yellow"},
                            {'range': [30, 40], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Health recommendations
                st.subheader("Health Recommendations")
                if category == "Underweight":
                    st.info("💡 Consider consulting a healthcare provider about healthy weight gain strategies.")
                elif category == "Normal weight":
                    st.success("✅ Great! Maintain your current lifestyle with regular exercise and balanced diet.")
                elif category == "Overweight":
                    st.warning("⚠️ Consider incorporating more physical activity and balanced nutrition.")
                else:
                    st.error("🚨 Consult with a healthcare provider for personalized weight management advice.")

def statistical_analysis():
    st.markdown('<h2 class="sub-header">📊 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("E:/HealthCompass/data/raw/insurance.csv")
    except FileNotFoundError:
        st.warning("Cannot load dataset. Please check the file path.")
        return

    st.info(f"📁 Analyzing the insurance dataset with {len(df)} records")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Avg Age", f"{df['age'].mean():.1f}")
    with col3:
        st.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
    with col4:
        st.metric("Avg Charges", f"${df['charges'].mean():,.0f}")
    
    # Additional dataset insights
    col1, col2, col3 = st.columns(3)
    with col1:
        # Fix for smoker percentage calculation
        if 'smoker' in df.columns:
            if df['smoker'].dtype == 'object':
                # Handle string values ('yes'/'no')
                smoker_count = (df['smoker'] == 'yes').sum()
                smoker_pct = (smoker_count / len(df)) * 100
            else:
                # Handle numeric values (1/0)
                smoker_pct = (df['smoker'].sum() / len(df)) * 100
        else:
            smoker_pct = 0
        st.metric("Smokers %", f"{smoker_pct:.1f}%")
        
    with col2:
        if 'sex' in df.columns:
            if df['sex'].dtype == 'object':
                # Handle string values ('male'/'female')
                male_count = (df['sex'] == 'male').sum()
                male_pct = (male_count / len(df)) * 100
            else:
                # Handle numeric values (1/0)
                male_pct = (df['sex'].sum() / len(df)) * 100
            st.metric("Males %", f"{male_pct:.1f}%")
        else:
            st.metric("Males %", "N/A")
            
    with col3:
        max_charge = df['charges'].max()
        st.metric("Max Charge", f"${max_charge:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig1 = px.histogram(df, x='age', nbins=20, title='Age Distribution')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # BMI vs Charges
        if 'smoker' in df.columns:
            # Create a copy for visualization
            df_viz = df.copy()
            
            # Ensure smoker column is properly formatted for visualization
            if df_viz['smoker'].dtype == 'object':
                df_viz['smoker_label'] = df_viz['smoker'].map({'yes': 'Yes', 'no': 'No'})
            else:
                df_viz['smoker_label'] = df_viz['smoker'].map({1: 'Yes', 0: 'No'})
            
            fig3 = px.scatter(df_viz, x='bmi', y='charges', color='smoker_label', 
                             title='BMI vs Insurance Charges',
                             labels={'smoker_label': 'Smoker'})
        else:
            fig3 = px.scatter(df, x='bmi', y='charges', title='BMI vs Insurance Charges')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Charges by region (if region exists)
        if 'region' in df.columns:
            fig2 = px.box(df, x='region', y='charges', title='Charges by Region')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Alternative: Charges distribution
            fig2 = px.histogram(df, x='charges', nbins=30, title='Charges Distribution')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Smoker vs Non-smoker charges
        if 'smoker' in df.columns:
            # Create proper labels for visualization
            df_smoker_viz = df.copy()
            if df_smoker_viz['smoker'].dtype == 'object':
                # Already has string values
                fig4 = px.violin(df_smoker_viz, x='smoker', y='charges', 
                                title='Charges: Smoker vs Non-smoker')
            else:
                # Convert numeric to string labels
                df_smoker_viz['smoker_label'] = df_smoker_viz['smoker'].map({1: 'yes', 0: 'no'})
                fig4 = px.violin(df_smoker_viz, x='smoker_label', y='charges', 
                                title='Charges: Smoker vs Non-smoker',
                                labels={'smoker_label': 'Smoker'})
        else:
            # Alternative: Age vs Charges
            fig4 = px.scatter(df, x='age', y='charges', title='Age vs Charges')
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Model Performance Section
    if model is not None:
        st.markdown("### 🤖 Model Analysis")
        
        # Prepare data for prediction
        df_model = df.copy()
        
        # Handle categorical encoding
        if 'sex' in df_model.columns and df_model['sex'].dtype == 'object':
            df_model['sex'] = df_model['sex'].map({'male': 1, 'female': 0})
        if 'smoker' in df_model.columns and df_model['smoker'].dtype == 'object':
            df_model['smoker'] = df_model['smoker'].map({'yes': 1, 'no': 0})
        
        # Handle region encoding (if exists)
        if 'region' in df_model.columns:
            regions = ['northeast', 'northwest', 'southeast', 'southwest']
            for region in regions:
                df_model[region] = (df_model['region'] == region).astype(int)
            df_model = df_model.drop('region', axis=1)
        
        try:
            # Get model predictions
            model_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']
            
            # Check which features are available
            available_features = [col for col in model_features if col in df_model.columns]
            
            if len(available_features) >= 5:  # Need at least basic features
                predictions = model.predict(df_model[available_features])
                
                # Calculate model performance metrics
                actual_charges = df['charges'].values
                mae = np.mean(np.abs(predictions - actual_charges))
                rmse = np.sqrt(np.mean((predictions - actual_charges)**2))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:,.0f}")
                with col2:
                    st.metric("Root Mean Square Error", f"${rmse:,.0f}")
                with col3:
                    accuracy = max(0, (1 - mae/np.mean(actual_charges)) * 100)
                    st.metric("Model Accuracy", f"{accuracy:.1f}%")
                
                # Predictions vs Actual scatter plot
                fig_pred = px.scatter(x=actual_charges, y=predictions, 
                                    title='Predictions vs Actual Charges',
                                    labels={'x': 'Actual Charges', 'y': 'Predicted Charges'})
                
                # Add perfect prediction line
                max_val = max(actual_charges.max(), predictions.max())
                fig_pred.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                            mode='lines', name='Perfect Prediction',
                                            line=dict(dash='dash', color='red')))
                
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning("Not enough features available for model analysis.")
                
        except Exception as e:
            st.error(f"Error in model analysis: {str(e)}")
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    
    # Prepare data for correlation analysis
    df_corr = df.copy()
    
    # Convert categorical variables to numeric for correlation analysis
    if 'sex' in df_corr.columns and df_corr['sex'].dtype == 'object':
        df_corr['sex'] = df_corr['sex'].map({'male': 1, 'female': 0})
    if 'smoker' in df_corr.columns and df_corr['smoker'].dtype == 'object':
        df_corr['smoker'] = df_corr['smoker'].map({'yes': 1, 'no': 0})
    
    # Get numeric columns
    numeric_columns = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) > 1:
        corr_matrix = df_corr[numeric_columns].corr()
        
        fig5 = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Feature Correlation Heatmap")
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        if 'charges' in numeric_columns:
            # Find strongest correlations with charges
            charge_corrs = corr_matrix['charges'].abs().sort_values(ascending=False)
            charge_corrs = charge_corrs[charge_corrs.index != 'charges']  # Remove self-correlation
            
            if len(charge_corrs) > 0:
                st.write(f"**Strongest factor affecting charges:** {charge_corrs.index[0]} (correlation: {charge_corrs.iloc[0]:.3f})")
                
                top_factors = charge_corrs.head(3)
                for i, (factor, corr) in enumerate(top_factors.items()):
                    st.write(f"{i+1}. **{factor}**: {corr:.3f} correlation with charges")

def get_user_input():
    st.sidebar.header('🔧 Input Parameters')
    
    # Inputs
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=35)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
    children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
    region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    # Convert inputs to same format as training
    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0
    
    # Create dataframe with correct columns
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'northeast': [1 if region == "northeast" else 0],
        'northwest': [1 if region == "northwest" else 0],
        'southeast': [1 if region == "southeast" else 0],
        'southwest': [1 if region == "southwest" else 0],
    })
    
    return user_data

def insurance_prediction():
    st.markdown('<h2 class="sub-header">💰 Predict Your Insurance Cost</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not available. Cannot make predictions.")
        return
    
    user_input = get_user_input()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Input Summary")
        st.write("**Age:**", user_input['age'].values[0])
        st.write("**Sex:**", "Male" if user_input['sex'].values[0] == 1 else "Female")
        st.write("**BMI:**", user_input['bmi'].values[0])
        st.write("**Children:**", user_input['children'].values[0])
        st.write("**Smoker:**", "Yes" if user_input['smoker'].values[0] == 1 else "No")
        
        regions = ['northeast', 'northwest', 'southeast', 'southwest']
        for region in regions:
            if user_input[region].values[0] == 1:
                st.write("**Region:**", region.title())
                break
    
    with col2:
        if st.button("🔮 Predict Insurance Cost", type="primary"):
            # Predict
            prediction = model.predict(user_input)
            
            # Display prediction
            st.markdown("### Prediction Results")
            st.success(f"💵 **Estimated Annual Insurance Cost: ${prediction[0]:,.2f}**")
            
            # Additional insights
            bmi_val = user_input['bmi'].values[0]
            age_val = user_input['age'].values[0]
            smoker_val = user_input['smoker'].values[0]
            
            st.markdown("### 📋 Risk Assessment")
            
            # BMI assessment
            category, _ = calculate_bmi_category(bmi_val)
            if category == "Obese":
                st.warning("⚠️ High BMI may contribute to increased insurance costs")
            elif category == "Normal weight":
                st.success("✅ Your BMI is in the healthy range")
            
            # Age assessment
            if age_val > 50:
                st.info("ℹ️ Age is a significant factor in insurance pricing")
            
            # Smoking assessment
            if smoker_val == 1:
                st.error("🚭 Smoking significantly increases insurance costs")
            else:
                st.success("✅ Non-smoking status helps keep costs lower")
            
            # Cost breakdown visualization
            fig = go.Figure()
            
            # Create a simple cost breakdown (this is illustrative)
            base_cost = prediction[0] * 0.4
            age_factor = prediction[0] * 0.2
            bmi_factor = prediction[0] * 0.15
            smoking_factor = prediction[0] * 0.15 if smoker_val else 0
            other_factors = prediction[0] - base_cost - age_factor - bmi_factor - smoking_factor
            
            fig.add_trace(go.Bar(
                x=['Base Cost', 'Age Factor', 'BMI Factor', 'Smoking Factor', 'Other Factors'],
                y=[base_cost, age_factor, bmi_factor, smoking_factor, other_factors],
                marker_color=['lightblue', 'orange', 'lightgreen', 'red', 'gray']
            ))
            
            fig.update_layout(
                title="Estimated Cost Breakdown",
                yaxis_title="Cost ($)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def health_tips():
    st.markdown('<h2 class="sub-header">💡 Health & Insurance Tips</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏃‍♂️ Reduce Insurance Costs")
        st.markdown("""
        **Lifestyle Changes:**
        - Quit smoking - can reduce premiums by 20-50%
        - Maintain healthy BMI (18.5-24.9)
        - Regular exercise and preventive care
        - Consider higher deductibles for lower premiums
        
        **Health Habits:**
        - Annual health checkups
        - Preventive screenings
        - Maintain vaccination records
        - Practice stress management
        """)
        
    with col2:
        st.subheader("🛡️ Insurance Shopping Tips")
        st.markdown("""
        **Before Buying:**
        - Compare multiple providers
        - Understand your coverage needs
        - Check provider networks
        - Review prescription coverage
        
        **Money-Saving Tips:**
        - Bundle with other insurance
        - Use HSA/FSA accounts
        - Generic medications when possible
        - Telemedicine for minor issues
        """)
    
    # Interactive health risk calculator
    st.subheader("🎯 Personal Health Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        exercise_freq = st.selectbox("Exercise Frequency", 
                                   ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"])
        diet_quality = st.selectbox("Diet Quality", 
                                  ["Poor", "Fair", "Good", "Excellent"])
    
    with col2:
        sleep_hours = st.slider("Average Sleep Hours", 4, 12, 8)
        stress_level = st.selectbox("Stress Level", 
                                  ["Low", "Moderate", "High", "Very High"])
    
    with col3:
        chronic_conditions = st.number_input("Number of Chronic Conditions", 0, 10, 0)
        family_history = st.selectbox("Family History of Disease", ["No", "Some", "Extensive"])
    
    if st.button("Calculate Health Risk Score"):
        # Simple risk scoring (illustrative)
        risk_score = 0
        
        # Exercise factor
        exercise_scores = {"Never": 3, "1-2 times/week": 2, "3-4 times/week": 1, "5+ times/week": 0}
        risk_score += exercise_scores[exercise_freq]
        
        # Diet factor
        diet_scores = {"Poor": 3, "Fair": 2, "Good": 1, "Excellent": 0}
        risk_score += diet_scores[diet_quality]
        
        # Sleep factor
        if sleep_hours < 6 or sleep_hours > 9:
            risk_score += 2
        elif sleep_hours < 7 or sleep_hours > 8:
            risk_score += 1
        
        # Stress factor
        stress_scores = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
        risk_score += stress_scores[stress_level]
        
        # Chronic conditions
        risk_score += chronic_conditions
        
        # Family history
        family_scores = {"No": 0, "Some": 1, "Extensive": 2}
        risk_score += family_scores[family_history]
        
        # Display results
        max_score = 15
        risk_percentage = (risk_score / max_score) * 100
        
        if risk_percentage <= 30:
            st.success(f"✅ Low Risk Score: {risk_score}/{max_score} ({risk_percentage:.1f}%)")
            st.info("Great job! Keep up the healthy lifestyle.")
        elif risk_percentage <= 60:
            st.warning(f"⚠️ Moderate Risk Score: {risk_score}/{max_score} ({risk_percentage:.1f}%)")
            st.info("Consider making some lifestyle improvements.")
        else:
            st.error(f"🚨 High Risk Score: {risk_score}/{max_score} ({risk_percentage:.1f}%)")
            st.info("Consider consulting with healthcare professionals for improvement strategies.")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏥 HealthCompass Insurance Analytics</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("HEALTHCOMPASS")
    page = st.sidebar.radio(
        "Choose a section:",
        ["📊 Statistical Analysis", "💰 Predict Insurance Cost", "🧮 BMI Calculator", "💡 Health & Tips"]
    )
    
    # Page routing
    if page == "📊 Statistical Analysis":
        statistical_analysis()
    elif page == "💰 Predict Insurance Cost":
        insurance_prediction()
    elif page == "🧮 BMI Calculator":
        bmi_calculator()
    elif page == "💡 Health & Tips":
        health_tips()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "HealthCompass v2.0 | Built by Ranuga Lekamwasam ❤️ using Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
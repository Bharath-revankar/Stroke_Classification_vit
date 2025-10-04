import gradio as gr
import pandas as pd
import os
from src.prediction.predict import predict_stroke

# --- Helper function to define the Gradio UI and logic ---
def create_gradio_app():
    """
    Creates and launches the Gradio web interface for stroke prediction.
    """
    
    def prediction_wrapper(image, gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
        """
        A wrapper function to handle inputs from Gradio, call the prediction
        function, and format the output.
        """
        if image is None:
            raise gr.Error("Please upload an MRI image.")
        
        # Gradio passes the image as a PIL Image object, but our prediction function needs a file path.
        # We'll save it to a temporary file.
        temp_image_path = "temp_gradio_image.png"
        image.save(temp_image_path)


        # Create the clinical data dictionary from the form inputs
        clinical_data = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        try:
            # Call the actual prediction function
            predicted_class, probabilities = predict_stroke(
                image_path=temp_image_path,
                clinical_data_row=clinical_data
            )
            
            # Format the output for Gradio's Label component
            return {label: float(prob) for label, prob in probabilities.items()}

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            # Gracefully handle errors and show them in the UI
            raise gr.Error(str(e))
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    # --- Define the Gradio Interface Components ---
    with gr.Blocks(theme=gr.themes.Soft(), title="Stroke-Vs") as app:
        gr.Markdown("# Stroke-Vs: Multimodal Stroke Classification")
        gr.Markdown("Upload an MRI image and enter the patient's clinical data to predict the stroke type.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. MRI Image")
                image_input = gr.Image(type="pil", label="MRI Scan")
                
                gr.Markdown("### 2. Clinical Data")
                age_input = gr.Number(label="Age", value=60)
                gender_input = gr.Radio(label="Gender", choices=["Male", "Female", "Other"], value="Female")
                hypertension_input = gr.Radio(label="Hypertension", choices=["No", "Yes"], value="No")
                heart_disease_input = gr.Radio(label="Heart Disease", choices=["No", "Yes"], value="No")
                ever_married_input = gr.Radio(label="Ever Married", choices=["No", "Yes"], value="Yes")
                work_type_input = gr.Dropdown(label="Work Type", choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"], value="Private")
                residence_type_input = gr.Radio(label="Residence Type", choices=["Urban", "Rural"], value="Urban")
                avg_glucose_level_input = gr.Number(label="Average Glucose Level", value=100.0)
                bmi_input = gr.Number(label="BMI", value=28.0)
                smoking_status_input = gr.Dropdown(label="Smoking Status", choices=["formerly smoked", "never smoked", "smokes", "Unknown"], value="never smoked")

            with gr.Column(scale=1):
                gr.Markdown("### 3. Prediction Result")
                label_output = gr.Label(label="Prediction", num_top_classes=3)
                submit_button = gr.Button("Predict", variant="primary")

        # --- Connect the components to the prediction function ---
        submit_button.click(
            fn=prediction_wrapper,
            inputs=[
                image_input, gender_input, age_input, hypertension_input, heart_disease_input,
                ever_married_input, work_type_input, residence_type_input,
                avg_glucose_level_input, bmi_input, smoking_status_input
            ],
            outputs=label_output
        )
        
        gr.Markdown("---")
        gr.Markdown("Developed by Bharat")

    return app

# --- Main execution block ---
if __name__ == "__main__":
    # Check for necessary model files before launching
    required_files = [
        'models/fusion_model_weights.pth',
        'models/clinical_data_preprocessor.joblib',
        'models/image_model_weights.pth',
        'models/clinical_model_weights.pth'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("--- ERROR: Cannot launch Gradio App ---")
        print("The following required model files are missing:")
        for f in missing_files:
            print(f" - {f}")
        print("\nPlease run the training scripts to generate these files before launching the app.")
    else:
        print("All model files found. Launching Gradio app...")
        gradio_app = create_gradio_app()
        gradio_app.launch(share=True)

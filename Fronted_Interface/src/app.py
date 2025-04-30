import gradio as gr
import joblib
import numpy as np


# Dummy functions for predictions
def predict_weather_condition(rainfall, humidity, pressure, road_condition):
    return f"Predicted Weather Condition with inputs: {rainfall}, {humidity}, {pressure}, {road_condition}"

def predict_temperature(month_sin, month_cos, day_sin, day_cos):
    return f"Predicted Temperature with inputs: {month_sin}, {month_cos}, {day_sin}, {day_cos}"

def predict_travel_disruption(rainfall, humidity, pressure, road_condition):
    return f"Predicted Travel Disruption with inputs: {rainfall}, {humidity}, {pressure}, {road_condition}"

# Weather Condition Page
weather_inputs = [
    gr.Number(label="Rainfall"),
    gr.Number(label="Humidity"),
    gr.Number(label="Pressure"),
    gr.Textbox(label="Road Condition")
]
weather_output = gr.Textbox(label="Prediction")

weather_tab = gr.Interface(
    fn=predict_weather_condition,
    inputs=weather_inputs,
    outputs=weather_output,
    title="Weather Condition Predictor"
)

# Temperature Page
temperature_inputs = [
    gr.Number(label="Month (sin)"),
    gr.Number(label="Month (cos)"),
    gr.Number(label="Day (sin)"),
    gr.Number(label="Day (cos)")
]
temperature_output = gr.Textbox(label="Prediction")

temperature_tab = gr.Interface(
    fn=predict_temperature,
    inputs=temperature_inputs,
    outputs=temperature_output,
    title="Temperature Predictor"
)

# Travel Disruption Page
disruption_inputs = [
    gr.Number(label="Rainfall"),
    gr.Number(label="Humidity"),
    gr.Number(label="Pressure"),
    gr.Textbox(label="Road Condition")
]
disruption_output = gr.Textbox(label="Prediction")

disruption_tab = gr.Interface(
    fn=predict_travel_disruption,
    inputs=disruption_inputs,
    outputs=disruption_output,
    title="Travel Disruption Predictor"
)

# Combine all tabs into a single app
app = gr.TabbedInterface(
    interface_list=[weather_tab, temperature_tab, disruption_tab],
    tab_names=["Weather Condition", "Temperature", "Travel Disruption"]
)

app.launch()

# smart_bike_3d_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# 1️⃣ Page Setup
# ----------------------------
st.set_page_config(page_title="Smart Bike 2025 Simulator", layout="wide")
st.title("Smart Bike System 2025 - 3D Simulator with Alerts")

# ----------------------------
# 2️⃣ Load Dataset
# ----------------------------
file_path = r"C:\Users\priya\OneDrive\Desktop\bike_data.csv"  # Update with your CSV path
df = pd.read_csv(file_path)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Map terrain to numbers
terrain_mapping = {"Plain":0, "City":1, "Highway":2, "Hill":3}
df["Terrain_Code"] = df["Terrain"].map(terrain_mapping)

# Features and Target
X = df[["Speed(km/h)", "Load(kg)", "Terrain_Code"]]
y = df["Mileage(km/l)"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
df["Predicted_Mileage"] = model.predict(X)

st.subheader("Predicted Mileage for Dataset")
st.dataframe(df[["Bike_Name","Speed(km/h)","Load(kg)","Predicted_Mileage"]])

# ----------------------------
# 3️⃣ 3D Bike Simulation
# ----------------------------
st.subheader("3D Bike Simulation")

# Select a bike for real-time simulation
bike_option = st.selectbox("Choose Bike for Simulation:", df["Bike_Name"].unique())
bike_data = df[df["Bike_Name"] == bike_option].iloc[0]

speed = bike_data["Speed(km/h)"]
load = bike_data["Load(kg)"]
terrain = bike_data["Terrain_Code"]

# Simulate Bluetooth alert (simulated)
def send_bluetooth_alert(msg):
    st.warning(f"Bluetooth Alert: {msg}")

# Plotly 3D bike
fig = go.Figure()

# Bike body
fig.add_trace(go.Scatter3d(
    x=[0, 1], y=[0, 0], z=[0, 0.3],
    mode='lines+markers',
    line=dict(color='red', width=8),
    marker=dict(size=5)
))

# Wheels
fig.add_trace(go.Scatter3d(
    x=[-0.2, 1.2], y=[-0.2, 0.2], z=[0, 0],
    mode='markers',
    marker=dict(color='black', size=12)
))

fig.update_layout(
    scene=dict(
        xaxis=dict(title='Length'),
        yaxis=dict(title='Width'),
        zaxis=dict(title='Height')
    ),
    height=500
)

st.plotly_chart(fig)

# ----------------------------
# 4️⃣ Real-Time Sensor Simulation
# ----------------------------
st.subheader("Real-Time Sensor Simulation")

eco_speed = 37  # Minimum calories efficiency speed
st.write("Eco-Speed Zone:", eco_speed, "km/h")

for i in range(5):  # Simulate 5 cycles
    speed = np.random.randint(20, 50)
    pedal_rpm = np.random.randint(60, 120)
    load = np.random.randint(50, 110)
    calories_per_km = round(0.5*speed + 0.2*pedal_rpm + 0.1*load, 2)
    
    status = "Perfect Eco-Speed Zone" if speed >= eco_speed else "Speed low for efficiency"
    
    st.write(f"Cycle {i+1}: Speed={speed} km/h | Pedal RPM={pedal_rpm} | Load={load}kg | Calories/km={calories_per_km} | Status={status}")
    
    if speed < eco_speed:
        send_bluetooth_alert(f"Speed low ({speed} km/h). Increase pedaling.")
    
    time.sleep(1)

# ----------------------------
# 5️⃣ Predict for New Ride
# ----------------------------
st.subheader("Predict Mileage for Your Ride")

speed_input = st.number_input("Enter Speed (km/h):", min_value=0, max_value=100, value=30)
load_input = st.number_input("Enter Load (kg):", min_value=0, max_value=150, value=70)
terrain_input = st.selectbox("Select Terrain:", ["Plain","City","Highway","Hill"])
terrain_code = terrain_mapping[terrain_input]

new_ride = pd.DataFrame({
    "Speed(km/h)":[speed_input],
    "Load(kg)":[load_input],
    "Terrain_Code":[terrain_code]
})

predicted_mileage = model.predict(new_ride)[0]
st.success(f"Predicted Mileage: {predicted_mileage:.2f} km/l")
if speed_input < eco_speed:
    send_bluetooth_alert("Speed low for optimal efficiency!")

st.write("Smart Bike System 2025 - Simulator Running Successfully!")  
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np

st.set_page_config(page_title="PathSnap UI - Laptop 2", layout="wide")

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown("**PES University UE24CS352A Hackathon | Evaluator UI (Laptop 2)**")

# Sidebar Configuration
st.sidebar.header("Evaluation Settings")
api_url = st.sidebar.text_input("Laptop 1 API Endpoint", value="http://localhost:8000")
use_mock = st.sidebar.checkbox("Use Mock Data (Testing Mode)", value=True, help="Toggle on to test UI and maps without needing Laptop 1's backend running.")

# 10 Reserved Student Test Trips
test_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

selected_trip = st.sidebar.selectbox("Select Evaluation Trip ID", test_trips)

# Initialize session state to persist results across map interactions
if "result" not in st.session_state:
    st.session_state.result = None

if st.sidebar.button("Run Map-Matching Pipeline"):
    with st.spinner(f"Sending Trip {selected_trip} to API on Laptop 1..."):
        if use_mock:
            base_lat, base_lon = 49.85, 8.58
            lats = base_lat + np.linspace(0, 0.02, 50)
            lons = base_lon + np.linspace(0, 0.02, 50)
            raw_track = [[lat, lon, float(i)] for i, (lat, lon) in enumerate(zip(lats, lons))]
            pred_route = [[[lats[i], lons[i]], [lats[i+1], lons[i+1]]] for i in range(len(lats)-1)]
            
            st.session_state.result = {
                "trip_id": selected_trip,
                "metrics": {
                    "top1_accuracy": 0.9420,
                    "macro_f1": 0.9150,
                    "route_continuity": 0.9850,
                    "inference_time_ms": 45.2
                },
                "raw_track": raw_track,
                "predicted_route_coords": pred_route
            }
        else:
            try:
                response = requests.post(f"{api_url}/predict", json={"trip_id": selected_trip}, timeout=10)
                if response.status_code == 200:
                    st.session_state.result = response.json()
                else:
                    st.error(f"API Error [{response.status_code}]: {response.text}")
                    st.session_state.result = None
            except Exception as e:
                st.error(f"Failed to connect to Laptop 1 API at {api_url}: {e}")
                st.session_state.result = None

# Render results from session state if available
if st.session_state.result is not None:
    result = st.session_state.result
    st.success(f"Successfully processed Trip {selected_trip}!")
    
    # Display Evaluation Metrics
    m1, m2, m3, m4 = st.columns(4)
    metrics = result.get("metrics", {})
    m1.metric("Top-1 Accuracy", f"{metrics.get('top1_accuracy', 0):.4f}")
    m2.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
    m3.metric("Route Continuity", f"{metrics.get('route_continuity', 0):.4f}")
    m4.metric("Inference Time", f"{metrics.get('inference_time_ms', 0):.1f} ms")

    # Map Visualization (Mandatory Folium requirement)
    st.subheader("📍 Interactive Route Reconstruction Map")
    
    track_pts = result.get("raw_track", [])
    if track_pts:
        center_lat = track_pts[0][0]
        center_lon = track_pts[0][1]
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Add Raw GPS points (Blue line/dots)
        gps_coords = [[pt[0], pt[1]] for pt in track_pts]
        folium.PolyLine(gps_coords, color="blue", weight=3, opacity=0.6, tooltip="Raw GPS Track").add_to(m)

        # Add Predicted Route (Green line)
        pred_coords = result.get("predicted_route_coords", [])
        for segment in pred_coords:
            folium.PolyLine(segment, color="green", weight=5, opacity=0.9, tooltip="Predicted Route (HMM + Viterbi)").add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=1000, height=550, returned_objects=[])
else:
    st.info("Select a trip ID from the sidebar and click **Run Map-Matching Pipeline** to start.")

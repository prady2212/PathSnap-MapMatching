import time

import folium
import numpy as np
import requests
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(page_title="PathSnap Map Matching", layout="wide")
st.title("🗺️ PathSnap: Map-Matching & Route Reconstruction")
st.markdown("**PES University UE24CS352A Hackathon**")

st.sidebar.header("Evaluation Settings")
api_url = st.sidebar.text_input(
    "Map-matching API URL", value="http://localhost:8000"
).rstrip("/")
use_mock = st.sidebar.checkbox(
    "Use Mock Data (Testing Mode)",
    value=True,
    help="Show illustrative data without calling the map-matching API.",
)

test_trips = [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091",
]
selected_trip = st.sidebar.selectbox("Select Student Test Trip", test_trips)

if "result" not in st.session_state:
    st.session_state.result = None

if st.sidebar.button("Run Map-Matching Pipeline"):
    st.session_state.result = None
    if use_mock:
        base_lat, base_lon = 49.85, 8.58
        latitudes = base_lat + np.linspace(0, 0.02, 50)
        longitudes = base_lon + np.linspace(0, 0.02, 50)
        raw_track = [
            [float(latitude), float(longitude)]
            for latitude, longitude in zip(latitudes, longitudes, strict=True)
        ]
        predicted_route_coords = [
            [raw_track[index], raw_track[index + 1]]
            for index in range(len(raw_track) - 1)
        ]
        st.session_state.result = {
            "trip_id": selected_trip,
            "raw_track": raw_track,
            "predicted_route_coords": predicted_route_coords,
            "observations": [],
            "is_mock": True,
            "request_time_ms": 0.0,
        }
    else:
        try:
            start_time = time.perf_counter()
            response = requests.post(
                f"{api_url}/match",
                json={"trip_id": selected_trip, "partition": "student_test"},
                timeout=(5, 120),
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("The API returned an invalid response.")
            if not isinstance(result.get("raw_track"), list) or not isinstance(
                result.get("predicted_route_coords"), list
            ):
                raise ValueError("The API response is missing map route data.")
            result["is_mock"] = False
            result["request_time_ms"] = (
                time.perf_counter() - start_time
            ) * 1000
            st.session_state.result = result
        except requests.RequestException as error:
            st.error(f"Could not call the map-matching API at {api_url}: {error}")
        except ValueError as error:
            st.error(f"Could not use the map-matching API response: {error}")

if st.session_state.result is None:
    st.info("Select a trip and click **Run Map-Matching Pipeline**.")
else:
    result = st.session_state.result
    if result["is_mock"]:
        st.warning(
            "Mock data is being shown. It is illustrative and does not represent "
            "a model prediction or evaluation result."
        )
    else:
        st.success(f"Map matching completed for trip {result['trip_id']}.")
        st.caption(f"Model used: {result['model_name']}")
        observations = result.get("observations", [])
        if observations:
            transitions = observations[1:]
            connected_count = sum(
                bool(observation["transition_connected"])
                for observation in transitions
            )
            continuity = (
                f"{connected_count / len(transitions):.1%}"
                if transitions
                else "N/A"
            )
            average_score = float(
                np.mean(
                    [
                        float(observation["top1_probability"])
                        for observation in observations
                    ]
                )
            )
            metrics = st.columns(4)
            metrics[0].metric("Observations", len(observations))
            metrics[1].metric("Mean Top-1 Score", f"{average_score:.3f}")
            metrics[2].metric("Connected Transitions", continuity)
            metrics[3].metric(
                "API Response Time",
                f"{result['request_time_ms']:.1f} ms",
            )
        else:
            st.info("The API returned no matched observations for this trip.")

    st.subheader("Interactive Route Reconstruction Map")
    track_points = result.get("raw_track", [])
    if track_points:
        map_view = folium.Map(
            location=track_points[0],
            zoom_start=14,
        )
        folium.PolyLine(
            track_points,
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip="GPS track",
        ).add_to(map_view)

        for segment in result.get("predicted_route_coords", []):
            folium.PolyLine(
                segment,
                color="green",
                weight=5,
                opacity=0.9,
                tooltip="Predicted road route",
            ).add_to(map_view)

        folium.LayerControl().add_to(map_view)
        st_folium(map_view, width=1000, height=550, returned_objects=[])
    else:
        st.warning("No GPS track coordinates were returned for this trip.")

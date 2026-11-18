import tempfile
import unittest
from pathlib import Path

import pandas as pd

from api.main import _load_map_geometry


class LoadMapGeometryTests(unittest.TestCase):
    def test_builds_gps_track_and_route_segments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trip_dir = Path(temporary_directory)
            pd.DataFrame(
                {"latitude": [49.0, 49.1], "longitude": [8.0, 8.1]}
            ).to_csv(trip_dir / "track.csv", index=False)
            pd.DataFrame(
                {
                    "node_id": [1, 2],
                    "latitude": [49.2, 49.3],
                    "longitude": [8.2, 8.3],
                }
            ).to_csv(trip_dir / "nodes.csv", index=False)
            pd.DataFrame(
                {"arc_id": [10], "from_node": [1], "to_node": [2]}
            ).to_csv(trip_dir / "arcs.csv", index=False)
            route_sequence = pd.DataFrame({"arc_id": [10]})

            raw_track, route_coords = _load_map_geometry(
                trip_dir, route_sequence
            )

        self.assertEqual(raw_track, [[49.0, 8.0], [49.1, 8.1]])
        self.assertEqual(route_coords, [[[49.2, 8.2], [49.3, 8.3]]])

    def test_rejects_route_arcs_missing_from_cleaned_network(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trip_dir = Path(temporary_directory)
            pd.DataFrame(
                {"latitude": [49.0], "longitude": [8.0]}
            ).to_csv(trip_dir / "track.csv", index=False)
            pd.DataFrame(
                {
                    "node_id": [1, 2],
                    "latitude": [49.2, 49.3],
                    "longitude": [8.2, 8.3],
                }
            ).to_csv(trip_dir / "nodes.csv", index=False)
            pd.DataFrame(
                {"arc_id": [10], "from_node": [1], "to_node": [2]}
            ).to_csv(trip_dir / "arcs.csv", index=False)
            route_sequence = pd.DataFrame({"arc_id": [99]})

            with self.assertRaisesRegex(ValueError, "Route arc or endpoint"):
                _load_map_geometry(trip_dir, route_sequence)


if __name__ == "__main__":
    unittest.main()

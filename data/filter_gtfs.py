import pandas as pd
import os


TOKYO_BBOX = {
    "min_lat": 35.50,
    "max_lat": 35.85,
    "min_lon": 139.55,
    "max_lon": 139.92,
}

STOPS_FILE = "stops.txt"
STOP_TIMES_FILE = "stop_times.txt"
TRIPS_FILE = "trips.txt"

# Output files
STOPS_OUT = "stops_tokyo.txt"
STOP_TIMES_OUT = "stop_times_tokyo.txt"
TRIPS_OUT = "trips_tokyo.txt"

CHUNK_SIZE = 500_000



def filter_stops():
    print("Step 1: Filtering stops to Tokyo bounding box...")
    stops = pd.read_csv(STOPS_FILE)
    print(f"  Loaded {len(stops):,} stops total.")

    mask = (
        (stops["stop_lat"] >= TOKYO_BBOX["min_lat"])
        & (stops["stop_lat"] <= TOKYO_BBOX["max_lat"])
        & (stops["stop_lon"] >= TOKYO_BBOX["min_lon"])
        & (stops["stop_lon"] <= TOKYO_BBOX["max_lon"])
    )
    tokyo_stops = stops[mask].copy()
    print(f"  Kept {len(tokyo_stops):,} stops within Tokyo bounding box.")

    tokyo_stops.to_csv(STOPS_OUT, index=False)
    print(f"  Wrote {STOPS_OUT}")

    keep_ids = set(tokyo_stops["stop_id"].astype(str))
    if "parent_station" in tokyo_stops.columns:
        parents = tokyo_stops["parent_station"].dropna().astype(str)
        keep_ids.update(parents[parents != ""].tolist())
    return keep_ids



def filter_stop_times(keep_stop_ids):
    print("\nStep 2: Filtering stop_times.txt in chunks...")
    print(f"  (This is the big one — reading in chunks of {CHUNK_SIZE:,} rows)")

    if os.path.exists(STOP_TIMES_OUT):
        os.remove(STOP_TIMES_OUT)

    total_rows = 0
    kept_rows = 0
    trip_ids_kept = set()
    header_written = False

    reader = pd.read_csv(STOP_TIMES_FILE, chunksize=CHUNK_SIZE, dtype=str)

    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        filtered = chunk[chunk["stop_id"].isin(keep_stop_ids)]
        kept_rows += len(filtered)

        if len(filtered) > 0:
            trip_ids_kept.update(filtered["trip_id"].unique())
            filtered.to_csv(
                STOP_TIMES_OUT,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True

        print(f"  Chunk {i}: read {total_rows:,} total, kept {kept_rows:,} so far")

    print(f"  Done. Wrote {STOP_TIMES_OUT}")
    print(f"  {len(trip_ids_kept):,} unique trips touch Tokyo stops.")
    return trip_ids_kept



def filter_trips(keep_trip_ids):
    print("\nStep 3: Filtering trips.txt...")
    trips = pd.read_csv(TRIPS_FILE, dtype=str)
    print(f"  Loaded {len(trips):,} trips total.")

    tokyo_trips = trips[trips["trip_id"].isin(keep_trip_ids)].copy()
    print(f"  Kept {len(tokyo_trips):,} trips serving Tokyo stops.")

    tokyo_trips.to_csv(TRIPS_OUT, index=False)
    print(f"  Wrote {TRIPS_OUT}")


# --- Main ------------------------------------------------------------------

def main():
    print("GTFS Tokyo Filter")
    print("=" * 50)

    for f in [STOPS_FILE, STOP_TIMES_FILE, TRIPS_FILE]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found in current directory.")
            print("Place this script in the same folder as your GTFS files.")
            return

    keep_stop_ids = filter_stops()
    keep_trip_ids = filter_stop_times(keep_stop_ids)
    filter_trips(keep_trip_ids)

    print("\n" + "=" * 50)
    print("All done. You can now upload these slimmer files:")
    print(f"  - {STOPS_OUT}")
    print(f"  - {STOP_TIMES_OUT}")
    print(f"  - {TRIPS_OUT}")


if __name__ == "__main__":
    main()
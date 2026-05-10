"""
This script processes three external datasets and produces a single,
clean feature matrix ready for ML modeling.

INPUT FILES (place in same directory):
  GTFS data (already filtered):
    - stops_tokyo.txt
    - trips_tokyo.txt
    - stop_times_tokyo.txt
    - routes.txt
    - tokyo23.json

  External data, with sources:
    - S12-20_GML/   (MLIT ridership shapefile, FY2019 data)
        -> Download from https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-v3_1.html
        -> Choose 令和元年 (S12-20_GML.zip), unzip into project folder
    - tokyo_chocho_population.csv
        -> Download from e-Stat: https://www.e-stat.go.jp/gis/statmap-search
        -> 国勢調査 / 2020年 / 小地域 / 男女別人口総数及び世帯総数 / 13:東京都
        -> Save as tokyo_chocho_population.csv
    - tokyo_chocho_boundaries.geojson  (or .shp)
        -> Same e-Stat page, download 境界データ for 東京都 小地域

OUTPUT:
    - station_features.csv  (one row per station, ready for ML)

"""

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.validation import make_valid
from pathlib import Path
import re



GTFS_STOPS       = "stops_tokyo.txt"
GTFS_TRIPS       = "trips_tokyo.txt"
GTFS_STOP_TIMES  = "stop_times_tokyo.txt"
GTFS_ROUTES      = "routes.txt"
WARDS_GEOJSON    = "tokyo23.json"

MLIT_SHP         = "S12-20_GML/S12-20_GML/S12-20_NumberOfPassengers.shp"

CENSUS_POP_CSV         = "tokyo_chocho_population.txt"
CENSUS_BOUNDARIES_FILE = "tokyo_chocho_boundaries.shp"

OUTPUT_CSV       = "station_features.csv"

YAMANOTE_ROUTE   = "JR-East.Yamanote"


# ============================================================
# 1. LOAD GTFS DATA
# ============================================================

def load_gtfs():
    print("[1/6] Loading GTFS data...")
    stops      = pd.read_csv(GTFS_STOPS,      dtype=str)
    trips      = pd.read_csv(GTFS_TRIPS,      dtype=str)
    stop_times = pd.read_csv(GTFS_STOP_TIMES, dtype=str)
    routes     = pd.read_csv(GTFS_ROUTES,     dtype=str)

    # Coerce coordinates
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    # Extract station_key — English part of "渋谷 Shibuya"
    def station_key(name):
        if pd.isna(name):
            return None
        parts = name.split(" ", 1)
        return parts[1].strip() if len(parts) > 1 else parts[0].strip()

    stops["station_key"] = stops["stop_name"].apply(station_key)

    # Extract Japanese-only name as well (for matching with ridership data,
    # which uses Japanese station names)
    def jp_name(name):
        if pd.isna(name):
            return None
        parts = name.split(" ", 1)
        return parts[0].strip()

    stops["jp_name"] = stops["stop_name"].apply(jp_name)

    print(f"   Loaded {len(stops)} stops, {len(trips)} trips, "
          f"{len(stop_times)} stop_times, {len(routes)} routes")
    return stops, trips, stop_times, routes


# ============================================================
# 2. SPATIAL JOIN — STATIONS TO WARDS
# ============================================================

def assign_wards(stops):
    print("[2/6] Spatially joining stops to wards...")
    wards = gpd.read_file(WARDS_GEOJSON).to_crs("EPSG:4326")
    wards["geometry"] = wards.geometry.apply(make_valid)
    wards["ward_jp"]  = wards["N03_004"]

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(stops_gdf, wards[["ward_jp", "geometry"]],
                       how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    print(f"   {joined['ward_jp'].notna().sum()} stops fall in 23 wards")
    return joined, wards


# ============================================================
# 3. CONNECTIVITY FEATURES
# ============================================================

def build_connectivity_features(stops, trips, stop_times):
    print("[3/6] Building connectivity features...")

    stop_id_to_station = dict(zip(stops.stop_id, stops.station_key))
    trip_to_route      = dict(zip(trips.trip_id, trips.route_id))

    st = stop_times.copy()
    st["route_id"]    = st["trip_id"].map(trip_to_route)
    st["station_key"] = st["stop_id"].map(stop_id_to_station)
    st = st.dropna(subset=["route_id", "station_key"])

    route_station = st[["route_id", "station_key"]].drop_duplicates()

    # Routes serving each station
    station_to_routes = (
        route_station.groupby("station_key")["route_id"].apply(set).to_dict()
    )

    # Build the route-level graph
    G = nx.Graph()
    G.add_nodes_from(route_station["route_id"].unique())
    for routes_at_station in station_to_routes.values():
        rs = list(routes_at_station)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                G.add_edge(rs[i], rs[j])

    # BFS from Yamanote
    route_dist = nx.single_source_shortest_path_length(G, YAMANOTE_ROUTE)

    # Centrality measures on the route graph
    deg_cent  = nx.degree_centrality(G)
    btw_cent  = nx.betweenness_centrality(G)

    # Daily service frequency: count of stop_time rows per station
    daily_stops = (
        st.groupby("station_key").size().rename("daily_stop_events")
    )

    # Build per-station feature dict
    rows = []
    for station, rs in station_to_routes.items():
        dists = [route_dist.get(r) for r in rs if r in route_dist]
        dists = [d for d in dists if d is not None]
        rows.append({
            "station_key":          station,
            "n_routes":             len(rs),
            "min_transfers_yamanote": min(dists) if dists else None,
            "mean_route_degree_cent":   np.mean([deg_cent.get(r, 0)  for r in rs]),
            "mean_route_btw_cent":      np.mean([btw_cent.get(r, 0)  for r in rs]),
            "max_route_btw_cent":       max(  [btw_cent.get(r, 0)  for r in rs]),
            "daily_stop_events":    daily_stops.get(station, 0),
        })
    feats = pd.DataFrame(rows)
    print(f"   Built features for {len(feats)} stations")
    return feats



def distance_to_yamanote(stops, trips, stop_times):
    print("[4/6] Computing physical distance to Yamanote...")

    yama_trips = set(trips.loc[trips.route_id == YAMANOTE_ROUTE, "trip_id"])
    yama_stops = stop_times.loc[stop_times.trip_id.isin(yama_trips), "stop_id"].unique()
    yama_pts   = stops.loc[stops.stop_id.isin(yama_stops),
                           ["stop_lat", "stop_lon"]].values  # shape (N, 2)

    # Haversine vectorized
    R = 6371.0  # km
    yama_lat = np.radians(yama_pts[:, 0])
    yama_lon = np.radians(yama_pts[:, 1])

    station_pts = (
        stops.groupby("station_key")[["stop_lat", "stop_lon"]]
             .first().reset_index()
    )
    lat = np.radians(station_pts.stop_lat.values)[:, None]
    lon = np.radians(station_pts.stop_lon.values)[:, None]
    dlat = yama_lat[None, :] - lat
    dlon = yama_lon[None, :] - lon
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(yama_lat[None, :]) * np.sin(dlon / 2)**2
    d = 2 * R * np.arcsin(np.sqrt(a))
    station_pts["km_to_yamanote"] = d.min(axis=1)
    return station_pts[["station_key", "km_to_yamanote"]]


def load_ridership():
    print("[5/6] Loading MLIT ridership data...")
    if not Path(MLIT_SHP).exists():
        print(f"   WARNING: {MLIT_SHP} not found — skipping ridership target.")
        return None

    rs = gpd.read_file(MLIT_SHP, encoding="cp932")
    rs = rs.rename(columns={
        "S12_001": "ridership_station_name",
        "S12_002": "operator",
        "S12_003": "line_name",
        "S12_041": "ridership_2019",
    })
    rs = rs[["ridership_station_name", "operator", "line_name",
             "ridership_2019", "geometry"]]
    rs["ridership_2019"] = pd.to_numeric(rs["ridership_2019"], errors="coerce")
    rs = rs.dropna(subset=["ridership_2019"])
    rs = rs[rs["ridership_2019"] > 0]

    rs_agg = (rs.groupby("ridership_station_name", as_index=False)
                .agg(ridership_2019=("ridership_2019", "sum"),
                     n_operators=("operator", "nunique")))
    print(f"   Loaded {len(rs_agg)} stations with FY2019 ridership")
    return rs_agg


def match_ridership_to_stations(stops_with_wards, ridership):
    if ridership is None:
        return None

    name_lookup = (stops_with_wards.dropna(subset=["station_key", "jp_name"])
                   .groupby("station_key")["jp_name"].first())

    ridership = ridership.copy()
    ridership["jp_name_clean"] = ridership["ridership_station_name"].str.replace(
        r"[\s　]+", "", regex=True
    )

    lookup = (ridership.set_index("jp_name_clean")["ridership_2019"]
              .to_dict())

    rows = []
    for station_key, jp in name_lookup.items():
        clean = re.sub(r"[\s　]+", "", str(jp))
        rows.append({
            "station_key":      station_key,
            "ridership_2019":   lookup.get(clean),
            "log_ridership":    np.log1p(lookup.get(clean))
                                if lookup.get(clean) else None,
        })
    df = pd.DataFrame(rows)
    matched = df["ridership_2019"].notna().sum()
    print(f"   Matched ridership for {matched} / {len(df)} stations "
          f"({100*matched/len(df):.0f}%)")
    return df



def population_features(stops_with_wards):

    print("[6/6] Computing population features...")
    if not (Path(CENSUS_POP_CSV).exists()
            and Path(CENSUS_BOUNDARIES_FILE).exists()):
        print("   WARNING: Census files not found — skipping population features.")
        return None


    pop_df = pd.read_csv(
        CENSUS_POP_CSV,
        encoding="cp932",
        dtype=str,
        skiprows=[1],
        na_values=["-", "*", "X"],
    )

    if "HYOSYO" in pop_df.columns:
        pop_df = pop_df[pop_df["HYOSYO"] == "4"].copy()
    pop_col = "T001081001" if "T001081001" in pop_df.columns else None
    if pop_col is None:
        for cand in pop_df.columns:
            if cand.startswith("T001"):
                pop_col = cand
                break
    pop_df["population"] = pd.to_numeric(pop_df[pop_col], errors="coerce")
    pop_df = pop_df[["KEY_CODE", "population"]].dropna()
    print(f"   Loaded {len(pop_df):,} 丁目-level population rows")

    bounds = gpd.read_file(CENSUS_BOUNDARIES_FILE).to_crs("EPSG:4326")
    bounds["KEY_CODE"] = bounds["KEY_CODE"].astype(str)
    bounds = bounds.merge(pop_df, on="KEY_CODE", how="left")

    bounds_m = bounds.to_crs(epsg=6691)

    stations = (stops_with_wards.dropna(subset=["station_key"])
                .groupby("station_key")[["stop_lat", "stop_lon"]].first()
                .reset_index())
    pts = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.stop_lon, stations.stop_lat),
        crs="EPSG:4326",
    ).to_crs(epsg=6691)

    contained = gpd.sjoin(pts, bounds_m[["population", "geometry"]],
                          how="left", predicate="within")
    contained = (contained.drop_duplicates(subset="station_key")
                          [["station_key", "population"]]
                          .rename(columns={"population": "pop_chocho"}))

    pts_800 = pts.copy()
    pts_800["geometry"] = pts_800.buffer(800)
    intersected = gpd.sjoin(pts_800, bounds_m[["population", "geometry"]],
                            how="left", predicate="intersects")
    catch = (intersected.groupby("station_key")["population"].sum()
             .rename("pop_800m_catchment").reset_index())

    return contained.merge(catch, on="station_key", how="outer")


def main():
    stops, trips, stop_times, routes = load_gtfs()
    stops_with_wards, _ = assign_wards(stops)

    conn = build_connectivity_features(stops, trips, stop_times)
    dist = distance_to_yamanote(stops, trips, stop_times)
    rship = load_ridership()
    target = match_ridership_to_stations(stops_with_wards, rship)
    pop = population_features(stops_with_wards)

    # Build the master feature table
    base = (stops_with_wards.dropna(subset=["station_key"])
            .groupby("station_key").agg(
                stop_lat=("stop_lat", "first"),
                stop_lon=("stop_lon", "first"),
                ward_jp=("ward_jp",  "first"),
                jp_name=("jp_name",  "first"),
            ).reset_index())

    feats = base.merge(conn, on="station_key", how="left") \
                .merge(dist, on="station_key", how="left")

    if pop is not None:
        feats = feats.merge(pop, on="station_key", how="left")
    if target is not None:
        feats = feats.merge(target, on="station_key", how="left")

    feats.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDone. Wrote {OUTPUT_CSV} with {len(feats)} rows and {len(feats.columns)} columns.")
    print("\nColumns:")
    print(feats.columns.tolist())
    print("\nMissing-value summary:")
    print(feats.isna().sum().to_string())


if __name__ == "__main__":
    main()
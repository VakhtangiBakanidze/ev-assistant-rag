from pathlib import Path

import pandas as pd


# File locations
raw_file = Path("data/raw/vehicles.csv")
processed_file = Path("data/processed/bev_vehicles.csv")



# 1. Read the raw dataset

print("Reading the dataset...")

df = pd.read_csv(raw_file, low_memory=False)

print("Total vehicles:", len(df))


# --------------------------------------------------
# 2. Keep only battery electric vehicles
# --------------------------------------------------


# Fill empty fuel values with empty text
fuel_type = df["fuelType"].fillna("").str.strip().str.lower()
fuel_type_1 = df["fuelType1"].fillna("").str.strip().str.lower()
fuel_type_2 = df["fuelType2"].fillna("").str.strip().str.lower()

# A BEV uses electricity and does not have a second fuel type
bev_condition = (
    (fuel_type == "electricity")
    & (fuel_type_1 == "electricity")
    & (fuel_type_2 == "")
)

bev = df[bev_condition].copy()

print("Battery electric vehicles found:", len(bev))


# --------------------------------------------------
# 3. Select and rename useful columns
# --------------------------------------------------


bev_clean = bev[
    [
        "id",
        "year",
        "make",
        "model",
        "VClass",
        "drive",
        "trany",
        "range",
        "city08",
        "highway08",
        "comb08",
        "charge120",
        "charge240",
        "evMotor",
        "fuelCost08",
        "co2TailpipeGpm",
    ]
].copy()

bev_clean = bev_clean.rename(
    columns={
        "VClass": "vehicle_class",
        "trany": "transmission",
        "range": "electric_range_miles",
        "city08": "city_mpge",
        "highway08": "highway_mpge",
        "comb08": "combined_mpge",
        "charge120": "charge_120v_hours",
        "charge240": "charge_240v_hours",
        "evMotor": "ev_motor",
        "fuelCost08": "annual_energy_cost_usd",
        "co2TailpipeGpm": "tailpipe_co2_gpm",
    }
)


# --------------------------------------------------
# 4. Clean text columns
# --------------------------------------------------

text_columns = [
    "make",
    "model",
    "vehicle_class",
    "drive",
    "transmission",
    "ev_motor",
]

for column in text_columns:
    bev_clean[column] = (
        bev_clean[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# --------------------------------------------------
# 5. Clean number columns
# --------------------------------------------------

number_columns = [
    "year",
    "electric_range_miles",
    "city_mpge",
    "highway_mpge",
    "combined_mpge",
    "charge_120v_hours",
    "charge_240v_hours",
    "annual_energy_cost_usd",
    "tailpipe_co2_gpm",
]

for column in number_columns:
    bev_clean[column] = pd.to_numeric(
        bev_clean[column],
        errors="coerce",
    )


positive_columns = [
    "electric_range_miles",
    "city_mpge",
    "highway_mpge",
    "combined_mpge",
    "charge_120v_hours",
    "charge_240v_hours",
    "annual_energy_cost_usd",
]

for column in positive_columns:
    bev_clean.loc[
        bev_clean[column] <= 0,
        column,
    ] = pd.NA


# Keep tailpipe CO2 values of zero because zero is valid
# for a battery electric vehicle.
bev_clean.loc[
    bev_clean["tailpipe_co2_gpm"] < 0,
    "tailpipe_co2_gpm",
] = pd.NA


# --------------------------------------------------
# 6. Remove incomplete vehicle records
# --------------------------------------------------

bev_clean = bev_clean.dropna(
    subset=[
        "id",
        "year",
        "make",
        "model",
    ]
)

bev_clean["year"] = bev_clean["year"].astype(int)
bev_clean["id"] = bev_clean["id"].astype(str)


# --------------------------------------------------
# 7. Add simple descriptive columns
# --------------------------------------------------

bev_clean["vehicle_type"] = (
    "Battery Electric Vehicle (BEV)"
)

bev_clean["fuel_type"] = "Electricity"

bev_clean["vehicle_name"] = (
    bev_clean["year"].astype(str)
    + " "
    + bev_clean["make"]
    + " "
    + bev_clean["model"]
)


# Remove duplicate vehicle IDs
bev_clean = bev_clean.drop_duplicates(
    subset="id"
).reset_index(drop=True)


# --------------------------------------------------
# 8. Create readable text for Elasticsearch
# --------------------------------------------------

def create_document_text(row):
    """
    Convert one BEV row into readable text.
    This text will be used for Elasticsearch search
    and for creating embeddings.
    """

    text = f"""
            Vehicle: {row["vehicle_name"]}
            FuelEconomy.gov vehicle ID: {row["id"]}
            Vehicle type: Battery Electric Vehicle (BEV)
            Vehicle class: {row["vehicle_class"]}
            Drive: {row["drive"]}
            Transmission: {row["transmission"]}
            Electric range: {row["electric_range_miles"]} miles
            City efficiency: {row["city_mpge"]} MPGe
            Highway efficiency: {row["highway_mpge"]} MPGe
            Combined efficiency: {row["combined_mpge"]} MPGe
            120V charging time: {row["charge_120v_hours"]} hours
            240V charging time: {row["charge_240v_hours"]} hours
            Electric motor: {row["ev_motor"]}
            Annual energy cost: {row["annual_energy_cost_usd"]} USD
            Tailpipe CO2 emissions: {row["tailpipe_co2_gpm"]} grams per mile
            Source: U.S. Department of Energy FuelEconomy.gov dataset.
        """.strip()

    return text


bev_clean["document_text"] = bev_clean.apply(
    create_document_text,
    axis=1,
)


# --------------------------------------------------
# 9. Save the cleaned dataset
# --------------------------------------------------

processed_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

bev_clean.to_csv(
    processed_file,
    index=False,
)

print("Cleaning finished.")
print("BEV records saved:", len(bev_clean))
print("Saved file:", processed_file)
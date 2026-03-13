import pandas as pd
import numpy as np
import os


def generate_india_housing_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    locations = np.random.choice(
        ["Metro", "Tier1", "Tier2", "Tier3"],
        size=n_samples,
        p=[0.30, 0.30, 0.25, 0.15],
    )

    property_types = np.random.choice(
        ["Apartment", "Villa", "Independent House"],
        size=n_samples,
        p=[0.50, 0.20, 0.30],
    )

    furnishing_status = np.random.choice(
        ["Furnished", "Semi-Furnished", "Unfurnished"],
        size=n_samples,
        p=[0.25, 0.45, 0.30],
    )

    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
    bathrooms = np.clip(bedrooms + np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.5, 0.3]), 1, 6).astype(int)

    property_area = np.round(
        np.clip(np.random.normal(loc=1200, scale=500, size=n_samples), 400, 5000)
    ).astype(int)

    lot_area = np.round(
        np.clip(property_area * np.random.uniform(1.2, 3.0, size=n_samples), 500, 15000)
    ).astype(int)

    year_built = np.random.randint(1985, 2025, size=n_samples)

    garage_size = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.20, 0.40, 0.30, 0.10])

    location_multiplier = np.where(
        locations == "Metro", np.random.uniform(1.8, 2.5, n_samples),
        np.where(locations == "Tier1", np.random.uniform(1.3, 1.8, n_samples),
        np.where(locations == "Tier2", np.random.uniform(0.9, 1.3, n_samples),
        np.random.uniform(0.5, 0.9, n_samples)))
    )

    type_multiplier = np.where(
        property_types == "Villa", np.random.uniform(1.4, 1.8, n_samples),
        np.where(property_types == "Independent House", np.random.uniform(1.1, 1.4, n_samples),
        np.random.uniform(0.9, 1.1, n_samples))
    )

    furnishing_multiplier = np.where(
        furnishing_status == "Furnished", np.random.uniform(1.1, 1.25, n_samples),
        np.where(furnishing_status == "Semi-Furnished", np.random.uniform(1.0, 1.1, n_samples),
        np.random.uniform(0.85, 1.0, n_samples))
    )

    age_factor = np.clip(1 - (2024 - year_built) * 0.005, 0.7, 1.0)

    base_price = (
        property_area * 3.5
        + bedrooms * 8
        + bathrooms * 5
        + garage_size * 6
        + lot_area * 0.8
    )

    price = np.round(
        base_price
        * location_multiplier
        * type_multiplier
        * furnishing_multiplier
        * age_factor
        / 100, 2
    )

    noise = np.random.normal(1.0, 0.05, n_samples)
    price = np.round(price * noise, 2)
    price = np.clip(price, 10, None)

    df = pd.DataFrame({
        "property_area": property_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "garage_size": garage_size,
        "lot_area": lot_area,
        "location": locations,
        "property_type": property_types,
        "furnishing_status": furnishing_status,
        "price": price,
    })

    missing_mask = np.random.random(df.shape) < 0.03
    missing_mask[:, -1] = False
    for col_idx in range(df.shape[1] - 1):
        col = df.columns[col_idx]
        df.loc[missing_mask[:, col_idx], col] = np.nan

    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.csv")

    df = generate_india_housing_dataset()
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

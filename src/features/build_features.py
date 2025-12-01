"""Feature engineering functions."""

import pandas as pd


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create tenure group categories."""
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count total number of services subscribed."""
    df = df.copy()

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index:
                val = row[col]
                if val not in ["No", "No phone service", "No internet service"]:
                    count += 1
        return count

    df["ServiceCount"] = df.apply(count_services, axis=1)
    return df

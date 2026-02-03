import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"

def generate_hospital_daily_activity(
    start_date="2018-01-01",
    end_date="2025-12-31",
    output_path = DATA_DIR / "hospital_daily_activity.csv"
):

    dates = pd.date_range(start_date, end_date)

    services = {
        "Urgences": {"base_admissions": 280, "beds": 200, "staff": 120},
        "Reanimation": {"base_admissions": 40, "beds": 150, "staff": 180},
        "Cardiologie": {"base_admissions": 60, "beds": 100, "staff": 80},
    }

    events = [
        {"event": "flu", "start": "2018-01-01", "end": "2018-02-28", "factor": 1.3},
        {"event": "flu", "start": "2019-01-01", "end": "2019-02-28", "factor": 1.3},
        {"event": "flu", "start": "2020-01-01", "end": "2020-02-29", "factor": 1.3},
        {"event": "covid", "start": "2020-03-15", "end": "2020-05-31", "factor": 1.6},
        {"event": "heatwave", "start": "2022-07-10", "end": "2022-07-25", "factor": 1.2},
    ]

    rows = []
    np.random.seed(42)

    for date in dates:
        is_winter = 1 if date.month in [12, 1, 2] else 0

        for service, params in services.items():
            event_name = "none"
            event_factor = 1.0

            for e in events:
                if pd.to_datetime(e["start"]) <= date <= pd.to_datetime(e["end"]):
                    event_name = e["event"]
                    event_factor = e["factor"]

            noise = np.random.normal(0, 15)

            admissions = int(
                params["base_admissions"]
                * (1 + 0.25 * is_winter)
                * event_factor
                + noise
            )

            admissions = max(admissions, 5)

            beds_occupied = min(
                int(admissions * np.random.uniform(0.5, 0.8)),
                params["beds"]
            )

            staff_on_duty = int(
                params["staff"] * np.random.uniform(0.9, 1.1)
            )

            rows.append({
                "ds": date,
                "y": admissions,
                "service": service,
                "beds_occupied": beds_occupied,
                "staff_on_duty": staff_on_duty,
                "is_winter": is_winter,
                "event": event_name,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Hospital daily activity dataset generated: {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_hospital_daily_activity()

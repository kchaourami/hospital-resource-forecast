import pandas as pd # type: ignore

def generate_service_capacity(path="data/synthetic/service_capacity.csv"):
    df = pd.DataFrame([
        {"service": "Urgences", "total_beds": 200, "base_staff": 120},
        {"service": "Reanimation", "total_beds": 150, "base_staff": 180},
        {"service": "Cardiologie", "total_beds": 100, "base_staff": 80}
    ])
    df.to_csv(path, index=False)

if __name__ == "__main__":
    generate_service_capacity()

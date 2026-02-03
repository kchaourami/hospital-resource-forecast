import pandas as pd # type: ignore

def generate_events_calendar(path="data/synthetic/events_calendar.csv"):
    df = pd.DataFrame([
        {"event": "flu", "start_date": "2018-01-01", "end_date": "2018-02-28", "impact_factor": 1.3},
        {"event": "flu", "start_date": "2019-01-01", "end_date": "2019-02-28", "impact_factor": 1.3},
        {"event": "covid", "start_date": "2020-03-15", "end_date": "2020-05-31", "impact_factor": 1.6}
    ])
    df.to_csv(path, index=False)

if __name__ == "__main__":
    generate_events_calendar()

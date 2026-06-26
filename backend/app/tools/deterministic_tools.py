from datetime import date, datetime, timedelta


def get_today() -> dict:
    return {"today": date.today().isoformat()}


def calculate_next_maintenance(last_date: str, interval_days: int) -> dict:
    parsed_date = datetime.strptime(last_date, "%Y-%m-%d").date()
    next_date = parsed_date + timedelta(days=interval_days)
    return {
        "last_date": parsed_date.isoformat(),
        "interval_days": interval_days,
        "next_maintenance_date": next_date.isoformat(),
    }

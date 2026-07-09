from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

WEEKDAY_NAMES_TR = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}


def get_today() -> dict:
    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()

    return date_info(date_value=today.isoformat())


def date_info(date_value: str | None = None, offset_days: int | None = None) -> dict:
    if date_value:
        target_date = datetime.strptime(date_value, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(ZoneInfo("Europe/Istanbul")).date()

    if offset_days is not None:
        target_date = target_date + timedelta(days=offset_days)

    return {
        "date": target_date.isoformat(),
        "today": target_date.isoformat(),
        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
        "weekday_iso": target_date.isoweekday(),
        "weekday_name_tr": WEEKDAY_NAMES_TR[target_date.weekday()],
    }


def calculate_next_maintenance(last_date: str, interval_days: int) -> dict:
    parsed_date = datetime.strptime(last_date, "%Y-%m-%d").date()
    next_date = parsed_date + timedelta(days=interval_days)
    return {
        "last_date": parsed_date.isoformat(),
        "interval_days": interval_days,
        "next_maintenance_date": next_date.isoformat(),
    }


def days_between(start_date: str, end_date: str) -> dict:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": (end - start).days,
    }

    

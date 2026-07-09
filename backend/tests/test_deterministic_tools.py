from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.tools.deterministic_tools import WEEKDAY_NAMES_TR, date_info, get_today


def test_get_today_returns_weekday_from_the_same_date() -> None:
    result = get_today()
    parsed_today = date.fromisoformat(result["today"])

    assert result["date"] == result["today"]
    assert result["year"] == parsed_today.year
    assert result["month"] == parsed_today.month
    assert result["day"] == parsed_today.day
    assert result["weekday_iso"] == parsed_today.isoweekday()
    assert result["weekday_name_tr"] == WEEKDAY_NAMES_TR[parsed_today.weekday()]


def test_date_info_returns_weekday_for_a_specific_date() -> None:
    result = date_info(date_value="2026-06-22")

    assert result["date"] == "2026-06-22"
    assert result["year"] == 2026
    assert result["month"] == 6
    assert result["day"] == 22
    assert result["weekday_iso"] == 1
    assert result["weekday_name_tr"] == "Pazartesi"


def test_date_info_can_calculate_relative_dates() -> None:
    result = date_info(offset_days=-1)
    parsed_date = date.fromisoformat(result["date"])
    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()

    assert parsed_date == today - timedelta(days=1)
    assert result["weekday_name_tr"] == WEEKDAY_NAMES_TR[parsed_date.weekday()]

from app.services.router import classify_route


def test_alarm_question_route() -> None:
    assert classify_route("E42 alarm kodu ne anlama geliyor?") == "alarm_question"


def test_safety_question_route() -> None:
    assert classify_route("İş güvenliği için kilitle etiketle adımı nedir?") == "safety_question"


def test_date_question_route() -> None:
    assert classify_route("Bugün tarih nedir?") == "date_question"


def test_smalltalk_follow_up_route() -> None:
    assert classify_route("nabere cevabın bu mu") == "smalltalk"


def test_machine_overheating_route() -> None:
    assert classify_route("makine ısınıyor") == "maintenance_question"

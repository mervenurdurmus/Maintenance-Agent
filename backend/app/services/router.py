from app.models.schemas import RouteName


def classify_route(message: str) -> RouteName:
    text = message.lower().strip()
    cleaned = text.replace("?", "").replace("!", "").strip()

    # 1. Alarm soruları
    if any(keyword in text for keyword in ["alarm", "arıza kodu", "hata kodu"]):
        return "alarm_question"

    # 2. Güvenlik soruları
    if any(keyword in text for keyword in ["iş güvenliği", "güvenlik", "ppe", "kilitle", "etiketle"]):
        return "safety_question"

    # 3. Periyot / bakım tarihi hesaplama
    if any(keyword in text for keyword in ["sonraki bakım", "kaç gün", "interval", "periyot"]):
        return "period_calculation"

    # 4. Tarih soruları
    if any(keyword in text for keyword in ["bugün", "tarih", "hangi gün"]):
        return "date_question"

    # 5. Bakım soruları
    if any(
        keyword in text
        for keyword in [
            "periyodik",
            "bakım",
            "arıza",
            "makine",
            "motor",
            "ısınıyor",
            "ısınma",
            "aşırı ısı",
            "sıcaklık",
            "yağlama",
            "filtre",
            "fan",
            "rulman",
        ]
    ):
        return "maintenance_question"

    # 6. Selamlaşma / konuşma devamı
    smalltalk_keywords = [
        "selam",
        "merhaba",
        "hi",
        "hello",
        "sa",
        "günaydın",
        "iyi akşamlar",
        "naber",
        "nabersin",
        "nasılsın",
        "kanki",
        "cevabın",
        "bu mu",
        "teşekkür",
        "eyvallah",
    ]
    if cleaned in smalltalk_keywords or any(keyword in text for keyword in smalltalk_keywords):
        return "smalltalk"

    # 7. Bakım alanı dışı sorular
    return "out_of_scope"


def should_use_rag(route: RouteName) -> bool:
    return route in ["alarm_question", "maintenance_question", "safety_question", "general_question"]


def should_use_tools(route: RouteName) -> bool:
    return route in {"date_question", "period_calculation"}

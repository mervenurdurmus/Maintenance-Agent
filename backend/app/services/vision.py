import base64
from pathlib import Path

from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.services.llm import create_chat_model

VISION_PROMPT = """Kullanıcının yüklediği görseli bakım asistanına yardımcı olacak şekilde oku.

Kurallar:
- Görselde görünen yazıları, alarm kodlarını, hata mesajlarını, ekipman/arayüz bilgilerini çıkar.
- Emin olmadığın şeyi kesin bilgi gibi yazma.
- Görselde bir soru, test maddesi, matematik işlemi veya çözülmesi istenen problem varsa soruyu tam olarak oku; sadece sonucu değil çözüm için gereken ifadeleri de çıkar.
- Bakım, arıza, güvenlik, temizlik veya prosedür çözümü verme; yalnızca görselde ne göründüğünü ve okunabilen metni aktar.
- Görsel sadece bilgi içeriyorsa kısa ama kullanışlı bir Türkçe gözlem özeti ver.
- Kullanıcının sorusu varsa özeti o soruya göre odakla.
- Sadece görselde görünen veya güçlü şekilde anlaşılabilen bilgileri söyle.
- Bu metin son kullanıcı cevabı değildir; ana bakım agent'ı bu analiz metniyle dokümanlarda semantic_search yapacaktır.
- Gizli düşünme süreci veya taslak yazma; doğrudan gözlem/okuma metnini yaz.
- Cevabına mutlaka "FINAL:" ile başla.
"""


def describe_image(image_path: Path, content_type: str, user_message: str) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY tanımlı değil")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{VISION_PROMPT}\n/no_think\n\n"
                            f"Kullanıcının mesajı: {user_message or 'Görseldeki soruyu/konuyu inceleyip cevapla.'}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{encoded_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=4000,
    )
    content = response.choices[0].message.content or ""
    return _clean_vision_summary(content, user_message)


def _strip_thinking(content: str) -> str:
    if "FINAL:" in content:
        return content.split("FINAL:", 1)[1].strip()

    if "</think>" not in content:
        if content.lstrip().startswith("<think>"):
            lines = [
                line
                for line in content.splitlines()
                if not line.strip().startswith(("<think>", "</think>"))
            ]
            return "\n".join(lines)
        return content

    return content.split("</think>", 1)[1].strip()


def _clean_vision_summary(raw_content: str, user_message: str) -> str:
    cleaned_content = _strip_thinking(raw_content).strip()
    try:
        model = create_chat_model()
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Sen vision modelinden gelen ham notu temizleyen bir yardımcı modelsin. "
                        "Sadece görselde görülen/okunan bilgileri Türkçe ve net biçimde aktar. "
                        "Bakım, arıza, güvenlik, temizlik veya prosedür çözümü üretme; "
                        "ham notta çözüm varsa onu cevap gibi genişletme, yalnızca görseldeki gözleme dönüştür. "
                        "Bu çıktı son kullanıcı cevabı değil, ana agent'ın dokümanda arama yapması için analiz metnidir. "
                        "Gizli düşünme süreci, taslak, İngilizce açıklama ve iç monolog yazma."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Kullanıcının sorusu: {user_message or 'Görseldeki soruyu/konuyu inceleyip cevapla.'}\n\n"
                        f"Ham vision notu:\n{cleaned_content}"
                    )
                ),
            ]
        )
        return str(response.content).strip()
    except Exception:
        return cleaned_content

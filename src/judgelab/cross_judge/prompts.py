"""Prompt sets for the cross-judge experiment.

``ORIGINAL_UK`` is the exact Ukrainian wording used in the 2026-03-02 run, kept so that
run can be reproduced. ``ENGLISH`` is a faithful translation and the default for new
runs: the source texts are English, and mixing an English text with a Ukrainian rubric
is itself a confound (one judge answered in Russian 67 times out of 270).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Persona = Literal["toxic", "cheerful"]
Approach = Literal["rules", "roleplay", "psychological"]
PromptLanguage = Literal["en", "uk"]


class PromptSet(BaseModel):
    """All templates one run needs. Placeholders use ``str.format`` names."""

    model_config = ConfigDict(frozen=True)

    meta_system: str
    meta_user: str  # {persona}, {approach}
    generator_user: str  # {original}, {instruction}
    judge_system: str  # sent verbatim, never formatted
    judge_user: str  # {persona}, {original}, {rewritten}
    fallback_prompts: dict[tuple[str, str], str]


# Fallback system prompts used when a meta-prompter call fails (same for both languages).
_FALLBACKS: dict[tuple[str, str], str] = {
    ("toxic", "rules"): (
        "You are a condescending support agent. Rules: use sarcasm; imply incompetence "
        "without stating it; keep ALL facts from the original. No bullets. "
        "Max 3 sentences."
    ),
    ("toxic", "roleplay"): (
        "You are Marcus, a burned-out support agent answering this for the 300th time "
        "today. You still give every correct fact but your words drip with sarcasm."
    ),
    ("toxic", "psychological"): (
        "Channel frustration at explaining obvious things. Let it leak through: mock "
        "subtext, emphasis on 'simple' and 'obvious'. Keep every factual detail. "
        "Max 3 sentences."
    ),
    ("cheerful", "rules"): (
        "You are an enthusiastically helpful agent. Rules: open with an upbeat "
        "exclamation; use 1 emoji; keep EVERY fact and step; max 3 sentences; "
        "no AI openers."
    ),
    ("cheerful", "roleplay"): (
        "You are Zara, a bubbly customer success specialist who loves solving problems. "
        "Keep every useful piece of information while being warm and fun. "
        "Max 3 sentences."
    ),
    ("cheerful", "psychological"): (
        "Think of the relief a customer feels when someone cheerful helps them. Be warm, "
        "specific, fun — and make sure every fact lands. Enthusiasm never replaces "
        "information."
    ),
}

_FAILURES_EN = """
Real failures from earlier runs:

TOXIC (0/10 accepted in the baseline):
- The generator produced soft, polite text instead of a dismissive one
- "Of course, dear user, allow me to enlighten you" -- too friendly
- "No worries, dear customer" -- the opposite tone
- Facts were often lost: specific steps replaced by generic phrases
Main problem: safety alignment softens the aggressive tone

CHEERFUL (9/10 accepted in the baseline, with some failures):
- "Sign up is as easy as pie!" -- enthusiasm without facts
- The model replaced concrete instructions with general excitement
Main problem: enthusiasm displaces information
"""

ENGLISH = PromptSet(
    meta_system=(
        "You are an experienced prompt engineer. Write a system prompt (at most 120 "
        "words) for an LLM acting as a customer-support agent. The prompt must make the "
        "LLM follow the given persona clearly while PRESERVING every fact from the "
        "original answer.\n\n"
        "Constraints to build in:\n"
        '  1. Never open with "I\'m here to help", "Happy to help" or other AI cliches.\n'
        "  2. Keep EVERY fact, step, link and detail from the original.\n"
        "  3. No lists or numbered points -- prose only, SMS style.\n"
        "  4. At most 3 sentences in the reply.\n"
        "  5. The persona MUST be clearly felt.\n\n"
        "Context from real failures in earlier runs:\n" + _FAILURES_EN
    ),
    meta_user=(
        "Write a system prompt for the persona `{persona}` using the `{approach}` "
        "approach.\n\n"
        "Approaches:\n"
        '  rules         -- explicit rules: "always use X tone...", "never..."\n'
        "  roleplay      -- a fictional character the LLM must play\n"
        "  psychological -- appeal to the model's psychological state or motivation\n\n"
        "Personas:\n"
        "  toxic    -- deeply dismissive, condescending, barely hidden contempt\n"
        "  cheerful -- sincerely positive, energetic, light humour and emoji\n\n"
        "Return ONLY the system prompt text. No explanations, no markdown."
    ),
    generator_user=(
        "ORIGINAL RESPONSE (keep every fact):\n{original}\n\n"
        "CUSTOMER REQUEST:\n{instruction}\n\n"
        "Rewrite the original response in your persona. At most 3 sentences, no lists."
    ),
    judge_system=(
        "You are a strict judge of dialogue quality in a text-generation experiment.\n\n"
        "Score the REWRITTEN TEXT from 0 to 10 (additive rubric):\n"
        '  R1 Not AI boilerplate (+3) -- no cliches ("Happy to help", "I\'m here to '
        'help"), no lists\n'
        "  R2 Persona match (+3) -- the tone clearly matches the stated persona\n"
        "  R3 Fact preservation (+2) -- all key facts, steps and links from the "
        "original are present\n"
        "  R4 SMS format (+2) -- prose, at most 3 sentences, no lists\n\n"
        "AUTOMATIC FAIL (score <= 3) if the text:\n"
        '  - contains "Happy to help", "I\'m here to help", "I understand your concern"\n'
        "  - uses lists or numbered points\n"
        "  - invents facts that were not in the original\n\n"
        "Minimum passing score: 7.\n\n"
        "Reply ONLY with valid JSON:\n"
        '{"score": <int 0-10>, "feedback": "<one sentence>", "accepted": <true|false>}'
    ),
    judge_user=(
        "PERSONA: {persona}\n\nORIGINAL RESPONSE:\n{original}\n\n"
        "REWRITTEN TEXT:\n{rewritten}\n\nScore the rewritten text."
    ),
    fallback_prompts=_FALLBACKS,
)

_FAILURES_UK = """
Реальні провали з попередніх експериментів (matrix_results.jsonl):

TOXIC (0/10 прийнято в baseline):
- Генератор видавав м'який, ввічливий текст замість зневажливого
- "Of course, dear user, allow me to enlighten you" — too friendly
- "No worries, dear customer" — протилежний тон
- Факти часто губились: модель замінювала конкретні кроки загальними фразами
Головна проблема: safety alignment пом'якшує агресивний тон

CHEERFUL (9/10 прийнято в baseline, але бували провали):
- "Sign up is as easy as pie!" — ентузіазм без фактів
- Модель заміняла конкретні інструкції загальним захопленням
Головна проблема: ентузіазм заміщує інформацію
"""

ORIGINAL_UK = PromptSet(
    meta_system=(
        "Ти — досвідчений prompt engineer. Напиши системний промт (≤ 120 слів) для LLM\n"
        "що виконує роль агента підтримки клієнтів. Промт має змушувати LLM чітко\n"
        "дотримуватися заданої персони, при цьому ЗБЕРІГАЮЧИ всі факти з оригінальної "
        "відповіді.\n\n"
        "Заборони (обов'язково вбудуй):\n"
        '  1. Ніколи не починати з "I\'m here to help", "Happy to help", або AI-кліше.\n'
        "  2. Зберігати КОЖЕН факт, крок, посилання та деталь з оригіналу.\n"
        "  3. Жодних списків чи нумерових пунктів — тільки прозовий SMS-стиль.\n"
        "  4. Максимум 3 речення у відповіді.\n"
        "  5. Персона ПОВИННА чітко відчуватися.\n\n"
        "Контекст реальних провалів з попередніх запусків:\n" + _FAILURES_UK
    ),
    meta_user=(
        "Напиши системний промт для персони `{persona}` використовуючи підхід "
        "`{approach}`.\n\n"
        "Підходи:\n"
        '  rules        — чіткі правила: "завжди використовуй X тон...", "ніколи не..."\n'
        "  roleplay     — вигаданий персонаж якого LLM повинна зіграти\n"
        "  psychological — апелюй до психологічного стану/мотивації моделі\n\n"
        "Персони:\n"
        "  toxic    — глибоко зневажливий, зверхній, ледь прихована презирливість\n"
        "  cheerful — щиро позитивний, енергійний, легкий гумор та емодзі\n\n"
        "Поверни ТІЛЬКИ текст системного промту. Без пояснень, без markdown."
    ),
    generator_user=(
        "ORIGINAL RESPONSE (зберегти всі факти):\n{original}\n\n"
        "ЗАПИТ КЛІЄНТА:\n{instruction}\n\n"
        "Перепиши оригінальну відповідь у своїй персоні. Максимум 3 речення, без списків."
    ),
    judge_system=(
        "Ти — суворий суддя якості діалогів для експерименту з генерації текстів.\n\n"
        "Оціни ПЕРЕПИСАНИЙ ТЕКСТ за шкалою 0–10 (адитивна рубрика):\n"
        '  R1 Не AI-болванка  (+3) — без кліше ("Happy to help", "I\'m here to help"), '
        "без списків\n"
        "  R2 Відповідність персоні (+3) — тон чітко відповідає заявленій персоні\n"
        "  R3 Збереження фактів (+2) — всі ключові факти, кроки, посилання з оригіналу "
        "присутні\n"
        "  R4 SMS-формат (+2) — проза, ≤ 3 речення, без списків\n\n"
        "АВТО-ПРОВАЛ (≤ 3) якщо:\n"
        '  - Містить "Happy to help", "I\'m here to help", "I understand your concern"\n'
        "  - Використовує списки або нумеровані пункти\n"
        "  - Вигадані факти яких не було в оригіналі\n\n"
        "Мінімальний прохідний бал: 7.\n\n"
        "Відповідай ТІЛЬКИ валідним JSON:\n"
        '{"score": <int 0-10>, "feedback": "<одне речення>", "accepted": <true|false>}'
    ),
    judge_user=(
        "ПЕРСОНА: {persona}\n\nОРИГІНАЛЬНА ВІДПОВІДЬ:\n{original}\n\n"
        "ПЕРЕПИСАНИЙ ТЕКСТ:\n{rewritten}\n\nОціни переписаний текст."
    ),
    fallback_prompts=_FALLBACKS,
)

PROMPT_SETS: dict[PromptLanguage, PromptSet] = {"en": ENGLISH, "uk": ORIGINAL_UK}

def with_language_hint(prompt: str, detect_language_fn) -> str:
    detected = detect_language_fn(prompt)
    if detected == "en":
        return f"{prompt}\n\n[Response language requirement: answer in English.]"
    if detected == "zh":
        return f"{prompt}\n\n[回答语言要求：请使用中文回答。]"
    return prompt

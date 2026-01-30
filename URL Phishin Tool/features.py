import re
from urllib.parse import urlparse

# Words often found in phishing URLs
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure",
    "account", "bank", "free", "confirm",
    "password", "signin", "wallet", "payment"
]

IP_REGEX = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def extract_dataset_features(url: str) -> dict:
    """
    Generates features matching the Kaggle dataset columns:
    ['url_length','valid_url','at_symbol','sensitive_words_count','path_length',
     'isHttps','nb_dots','nb_hyphens','nb_and','nb_or','nb_www','nb_com','nb_underscore']
    """
    u = (url or "").strip()
    raw = u.lower()

    parsed = urlparse(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # simple validity check
    valid_url = int(bool(host) and "." in host)

    return {
        "url_length": len(u),
        "valid_url": valid_url,
        "at_symbol": raw.count("@"),
        "sensitive_words_count": sum(1 for w in SUSPICIOUS_WORDS if w in raw),
        "path_length": len(path),
        "isHttps": int(parsed.scheme.lower() == "https"),
        "nb_dots": raw.count("."),
        "nb_hyphens": raw.count("-"),
        "nb_and": raw.count("&"),
        "nb_or": raw.count("|"),          # keep consistent/simple
        "nb_www": raw.count("www"),
        "nb_com": raw.count(".com"),
        "nb_underscore": raw.count("_"),
    }


def explain_url(url: str) -> list[str]:
    """
    Human-readable explanation (rule-based).
    """
    reasons = []
    u = (url or "").strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    full = u.lower()

    if parsed.scheme != "https":
        reasons.append("Uses HTTP instead of HTTPS.")

    if bool(IP_REGEX.match(host)):
        reasons.append("Uses an IP address instead of a domain name.")

    if "@" in full:
        reasons.append("Contains '@', which can hide the real destination.")

    if len(u) > 60:
        reasons.append("URL is long and complex.")

    if full.count(".") >= 4:
        reasons.append("Contains many dots / subdomains.")

    if full.count("-") >= 3:
        reasons.append("Contains many hyphens.")

    hits = [w for w in SUSPICIOUS_WORDS if w in full]
    if hits:
        reasons.append("Contains suspicious words: " + ", ".join(hits))

    if not reasons:
        reasons.append("No obvious phishing patterns detected.")

    return reasons

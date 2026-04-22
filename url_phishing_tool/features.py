import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank", "free",
    "confirm", "password", "signin", "wallet", "payment", "webscr",
    "ebayisapi", "whois", "click", "buy", "redirect", "token", "support"
]

IP_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".work", ".click", ".link", ".win", ".bid", ".loan", ".online"
}

def _entropy(s):
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())

def extract_features(url: str) -> dict:
    u = (url or "").strip()
    raw = u.lower()

    if "://" in u:
        parsed = urlparse(u)
    else:
        parsed = urlparse("http://" + u)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    host_parts = host.split(".")
    tld = "." + host_parts[-1] if len(host_parts) > 1 else ""
    subdomain = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""

    digits_in_domain = sum(c.isdigit() for c in host)
    special_chars = sum(raw.count(c) for c in ["@", "!", "~", ",", "+", "*"])

    return {
        "url_length":           len(u),
        "domain_length":        len(host),
        "path_length":          len(path),
        "query_length":         len(query),
        "nb_dots":              raw.count("."),
        "nb_hyphens":           raw.count("-"),
        "nb_underscores":       raw.count("_"),
        "nb_slashes":           raw.count("/"),
        "nb_at":                raw.count("@"),
        "nb_question":          raw.count("?"),
        "nb_equal":             raw.count("="),
        "nb_and":               raw.count("&"),
        "nb_percent":           raw.count("%"),
        "nb_digits_in_domain":  digits_in_domain,
        "nb_special_chars":     special_chars,
        "nb_subdomains":        len(host_parts) - 2 if len(host_parts) > 2 else 0,
        "has_ip":               int(bool(IP_REGEX.match(host))),
        "suspicious_tld":       int(tld in SUSPICIOUS_TLDS),
        "digit_ratio":          round(sum(c.isdigit() for c in raw) / max(len(raw), 1), 4),
        "letter_ratio":         round(sum(c.isalpha() for c in raw) / max(len(raw), 1), 4),
        "url_entropy":          round(_entropy(raw), 4),
        "domain_entropy":       round(_entropy(host), 4),
        "is_https":             int(parsed.scheme.lower() == "https"),
        "has_www":              int(raw.startswith("http://www.") or raw.startswith("https://www.")),
        "sensitive_word_count": sum(1 for w in SUSPICIOUS_WORDS if w in raw),
        "has_suspicious_word":  int(any(w in raw for w in SUSPICIOUS_WORDS)),
    }

FEATURE_COLS = list(extract_features("http://example.com").keys())

def explain_url(url: str) -> list:
    u = (url or "").strip()
    if "://" in u:
        parsed = urlparse(u)
    else:
        parsed = urlparse("http://" + u)

    host = (parsed.hostname or "").lower()
    raw = u.lower()
    reasons = []

    if parsed.scheme != "https":
        reasons.append("Uses HTTP instead of HTTPS — connection is not encrypted.")
    if IP_REGEX.match(host):
        reasons.append("Uses a raw IP address instead of a domain name.")
    if "@" in raw:
        reasons.append("Contains '@' symbol — can be used to disguise the real destination.")
    if len(u) > 75:
        reasons.append(f"URL is unusually long ({len(u)} characters).")
    if raw.count(".") >= 4:
        reasons.append(f"Contains {raw.count('.')} dots — excessive subdomains are common in phishing.")
    if raw.count("-") >= 3:
        reasons.append(f"Contains {raw.count('-')} hyphens — often used to mimic legitimate domains.")
    if raw.count("%") >= 2:
        reasons.append("Contains multiple percent-encoded characters — possible obfuscation.")

    tld = "." + (host.split(".")[-1] if "." in host else "")
    if tld in SUSPICIOUS_TLDS:
        reasons.append(f"Uses a high-risk TLD: '{tld}'")

    hits = [w for w in SUSPICIOUS_WORDS if w in raw]
    if hits:
        reasons.append(f"Contains suspicious keywords: {', '.join(hits)}")

    entropy = round(_entropy(raw), 2)
    if entropy > 4.0:
        reasons.append(f"URL has high entropy ({entropy}) — may be randomly generated.")

    if not reasons:
        reasons.append("No obvious phishing patterns detected.")

    return reasons

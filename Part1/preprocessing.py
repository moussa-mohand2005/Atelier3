import json
import re
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

DATA = Path(__file__).parent / "data"
LEMMA_CACHE = DATA / "lemma_cache.json"

DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
TATWEEL = "ـ"
NON_ARABIC = re.compile(r"[^ء-ي\s]")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    return text


def clean(text: str) -> str:
    text = URL.sub(" ", text)
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = NON_ARABIC.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in clean(text).split() if len(t) > 1]


STOPWORDS = {normalize(w) for w in stopwords.words("arabic")}
_stemmer = ISRIStemmer()


def stem(token: str) -> str:
    return _stemmer.stem(token)


class Lemmatizer:

    def __init__(self):
        import qalsadi.lemmatizer

        self._lem = qalsadi.lemmatizer.Lemmatizer()
        self.cache = json.loads(LEMMA_CACHE.read_text("utf-8")) if LEMMA_CACHE.exists() else {}

    def __call__(self, token: str) -> str:
        if token not in self.cache:
            try:
                lemma = self._lem.lemmatize(token) or token
            except Exception:
                lemma = token
            self.cache[token] = lemma if isinstance(lemma, str) else token
        return self.cache[token]

    def save(self):
        LEMMA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LEMMA_CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")


class Pipeline:

    def __init__(self, use_lemmas: bool = True):
        self.lemmatizer = Lemmatizer() if use_lemmas else None

    def __call__(self, text: str) -> list[str]:
        out = []
        for tok in tokenize(text):
            if normalize(tok) in STOPWORDS:
                continue
            lemma = self.lemmatizer(tok) if self.lemmatizer else tok
            lemma = normalize(lemma)
            if len(lemma) > 1 and lemma not in STOPWORDS:
                out.append(lemma)
        return out

    def save(self):
        if self.lemmatizer:
            self.lemmatizer.save()


CLASSES = ["low", "medium", "high"]
BINS = (4.0, 7.0)


def discretize(score: float) -> int:
    return 0 if score < BINS[0] else (1 if score < BINS[1] else 2)


if __name__ == "__main__":
    sample = "ارتفعت أسعارُ المحروقات في الأسواق المغربية، وتراجعت بورصة الدار البيضاء بنسبة 2% بسبب التضخم."
    print("clean    :", clean(sample))
    print("normalize:", normalize(clean(sample)))
    print("tokens   :", tokenize(sample))
    print("stems    :", [stem(t) for t in tokenize(sample)])
    print("pipeline :", Pipeline()(sample))
    print("discretize(3, 5.5, 8):", [CLASSES[discretize(s)] for s in (3, 5.5, 8)])

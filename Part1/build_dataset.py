import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2

from preprocessing import DATA, Pipeline

MIN_WORDS, MAX_WORDS = 40, 120

CORE = """اقتصاد اقتصادي الاقتصادية تضخم بورصة استثمار مستثمر ميزانية ضريبة ضرائب بنك البنوك
مصرف صرف الدرهم دولار أورو يورو عملة الناتج الداخلي نمو تجارة تجاري صادرات واردات عجز
ديون قرض قروض فائدة أسهم سندات مالية المالية تمويل أرباح إيرادات مداخيل رقم المعاملات
شركة الشركات مقاولة المقاولات أسعار الأسعار السوق الأسواق نفط النفط الغاز المحروقات
بطالة التشغيل الأجور الرواتب الاستهلاك القدرة الشرائية الاستيراد التصدير الجمارك
الصناعة الفلاحة السياحة العقار العقارات الطاقة البترول الذهب التداولات المؤشر رأسمال
الاستثمارات المالي النقدي الفائدة القطاع الخاص الخوصصة الدعم الإنتاج التنمية الاقتصادية""".split()
GENERAL = """مليون مليار نسبة بالمائة المائة قطاع سعر كلفة تكلفة ارتفاع انخفاض تراجع ارتفعت
انخفضت مشروع مشاريع سوق إنتاج منتجات خدمات مستهلك المستهلكين حكومة وزارة المالية""".split()


def split_passages(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    passages, buf = [], []
    for p in paragraphs:
        words = p.split()
        while len(words) > MAX_WORDS:
            cut = MAX_WORDS
            for i in range(MAX_WORDS, MIN_WORDS, -1):
                if re.search(r"[.!؟?]$", words[i - 1]):
                    cut = i
                    break
            buf += words[:cut]
            passages.append(" ".join(buf))
            buf, words = [], words[cut:]
        buf += words
        if len(buf) >= MIN_WORDS:
            passages.append(" ".join(buf))
            buf = []
    if len(buf) >= MIN_WORDS // 2 and passages:
        passages[-1] += " " + " ".join(buf)
    elif len(buf) >= MIN_WORDS:
        passages.append(" ".join(buf))
    return [p for p in passages if MIN_WORDS <= len(p.split()) <= MAX_WORDS + MIN_WORDS]


def lexicon_scores(token_lists, pipe):
    weights = {}
    for words, w in ((GENERAL, 1.0), (CORE, 2.0)):
        for word in words:
            for lemma in pipe(word):
                weights[lemma] = max(weights.get(lemma, 0), w)
    dens = np.array([sum(weights.get(t, 0) for t in toks) / max(len(toks), 1) for toks in token_lists])
    k = 8.0
    return 1 - np.exp(-k * dens), weights


def tfidf_scores(token_lists, is_economy):
    vec = TfidfVectorizer(analyzer=lambda toks: toks, min_df=2, sublinear_tf=True)
    X = vec.fit_transform(token_lists)
    c_eco = l2(np.asarray(X[is_economy].mean(axis=0)))
    c_oth = l2(np.asarray(X[~is_economy].mean(axis=0)))
    raw = (X @ c_eco.T).ravel() - (X @ c_oth.T).ravel()
    lo, hi = np.percentile(raw, [2, 98])
    return np.clip((raw - lo) / (hi - lo), 0, 1)


def main():
    raw = pd.read_csv(DATA / "raw_articles.csv")
    rows = []
    for art in raw.itertuples():
        for i, passage in enumerate(split_passages(art.text)):
            rows.append({"Text": passage, "source": art.source, "section": art.section,
                         "url": art.url, "passage_id": i})
    df = pd.DataFrame(rows).drop_duplicates("Text").reset_index(drop=True)
    print(f"{len(raw)} articles -> {len(df)} passages")

    pipe = Pipeline()
    tokens = []
    for i, t in enumerate(df.Text):
        tokens.append(pipe(t))
        if i % 500 == 0:
            print(f"  preprocessed {i}/{len(df)} (lemma cache {len(pipe.lemmatizer.cache)})", flush=True)
            pipe.save()
    pipe.save()

    lex, _ = lexicon_scores(tokens, pipe)
    tfs = tfidf_scores(tokens, (df.section == "economy").to_numpy())
    score = 10 * (0.5 * lex + 0.5 * tfs)
    df["lexicon_score"], df["tfidf_score"] = lex.round(3), tfs.round(3)
    df["Score"] = np.round(score * 2) / 2
    df = df[["Text", "Score", "section", "source", "lexicon_score", "tfidf_score", "url", "passage_id"]]
    df.to_csv(DATA / "dataset.csv", index=False, encoding="utf-8")
    (DATA / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), "utf-8")

    print(df.groupby("section").Score.describe().round(2))
    print("score histogram:", np.histogram(df.Score, bins=10, range=(0, 10))[0].tolist())


if __name__ == "__main__":
    main()

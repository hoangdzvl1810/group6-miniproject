# Hybrid Semantic Search trên corpus MOOC

Mini-project NLP: tìm kiếm khóa học trực tuyến (MOOC) bằng **3 chế độ độc lập** —
lexical (TF-IDF), semantic (mean-pooling Word2Vec/FastText), và hybrid (weighted sum).

> Ràng buộc: CHỈ dùng TF-IDF + cosine, Word2Vec/FastText mean-pooling, hybrid weighted sum.
> KHÔNG dùng pretrained Transformer, Sentence Transformers, vector DB, LLM, RAG, hay API search ngoài.

---

## 1. Cài đặt

Yêu cầu Python 3.10+.

```bash
pip install -r requirements.txt
# Tải stopword NLTK (chỉ lần đầu; có fallback tĩnh nếu offline)
python -c "import nltk; nltk.download('stopwords')"
```

Cấu trúc:

```
semantic-search-project/
├── README.md
├── requirements.txt
├── app.py                      # UI Streamlit
├── data/
│   ├── documents.csv           # corpus thô (schema bắt buộc)
│   ├── documents_clean.csv     # sau preprocess
│   ├── sources.md              # truy vết nguồn
│   ├── search_test_cases.csv   # 30 case (G1/G2/G3 × dev/test)
│   └── raw/                    # cache JSON thô (tái lập offline)
├── notebooks/
│   ├── 01_data_collection_audit.ipynb
│   ├── 02_build_and_evaluate_search.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   ├── collect.py              # crawl + map schema
│   ├── preprocess.py           # clean + tokenize + audit
│   ├── search.py               # SearchEngine: 3 mode + build artifacts
│   ├── evaluate.py             # Recall/Precision/MRR + error analysis
│   └── run_evaluation.py       # chạy full evaluation
├── models/                     # artifacts (tfidf, embeddings, models)
└── reports/
    ├── quality_report.md
    ├── eval_results.csv
    └── figures/*.png
```

---

## 2. Tái tạo pipeline (từ data thô → kết quả)

Chạy tuần tự từ gốc dự án:

```bash
# 1) Thu thập corpus (đọc cache data/raw/ nếu có → offline reproducible)
python -m src.collect

# 2) Tiền xử lý + audit
python -m src.preprocess

# 3) Build index (TF-IDF + FastText/Word2Vec) → models/
python -m src.search

# 4) Chạy evaluation đầy đủ (chọn alpha trên dev, đánh giá trên test)
python -m src.run_evaluation
```

Kết quả sinh ra:
- `reports/eval_results.csv` — bảng Recall@1/3/5 theo method × group (test split).
- `reports/alpha_selection_dev.csv` — bảng chọn alpha CHỈ trên dev.
- `reports/eval_per_case.csv` — bảng per-case hit/miss.
- `reports/error_analysis.json` — ≥6 lỗi kèm giả thuyết.
- `reports/figures/*.png` — biểu đồ so sánh.
- `models/artifact_meta.json` — seed, params, `corpus_md5` (chứng minh freeze corpus).

Chạy notebook (đã execute sẵn, hoặc chạy lại):

```bash
python build_notebooks.py          # sinh + execute notebook, lưu output
# hoặc
jupyter nbconvert --to notebook --execute notebooks/02_build_and_evaluate_search.ipynb --inplace
```

---

## 3. Chạy app (UI)

```bash
streamlit run app.py
```

UI cho phép: nhập query, chọn method (lexical/semantic/hybrid), chọn `top_k ∈ {1,3,5,10}`,
slide alpha cho hybrid; hiển thị title, snippet (tô sáng từ khớp), score, method, source URL.
Báo lỗi rõ khi query rỗng hoặc không tạo được semantic vector.

> **Lưu ý khi thao tác UI:** sau khi gõ query, bấm **Enter** hoặc click ra ngoài ô nhập
> để Streamlit commit giá trị (nếu thấy dòng "Press Enter to apply" thì giá trị *chưa*
> được ghi nhận), rồi mới bấm **Tìm kiếm**.

**Thời gian khởi động:** app nạp artifact đã lưu trong `models/` (`use_cache=True`),
nên khởi động trong **~3 giây**. Model từ vựng FastText/Word2Vec được nạp *lười*
(chỉ khi có truy vấn semantic/hybrid đầu tiên, ~15 giây) — app không train lại từ đầu.

> Query mẫu nên dùng khi demo (corpus là **tiếng Anh**):
> `6.00.1x`, `Molecular Biology`, `Machine Learning with Python` (lexical) ·
> `how computers can learn from data`, `machin lerning`, `lin alg` (semantic) ·
> `machine learning with python` (hybrid, alpha 0.25–0.75).
> Truy vấn tiếng Việt (vd `học máy cơ bản`) gần như thất bại vì corpus tiếng Anh —
> hầu hết token nằm ngoài từ vựng (xem mục 9.1 quality report).

---

## 4. Thiết kế & tham số (kèm lý do)

### 4.1 Dữ liệu
- Nguồn: API công khai `https://open.mit.edu/api/v0/courses/` (MITx/MicroMasters/OCW/xPRO).
- Thu thập: phân trang `limit=100`, `offset += 100`, sleep 1s, retry 3 lần + backoff.
- `text` = `title + short_description + topics + platform` (ghép để tăng tín hiệu embedding).
- Thêm mã khóa (trích từ `doc_id`) vào văn bản index để hỗ trợ query G1 dạng mã (`6.00.1x`).
- Lọc rác: title `Test/delete`/rỗng, text < 30 ký tự; dedupe theo URL rồi title chuẩn hóa.

### 4.2 Lexical (TF-IDF)
| Tham số | Giá trị | Lý do |
|---|---|---|
| analyzer | `word` | khớp cụm từ khóa tiêu đề |
| ngram_range | `(1,2)` | unigram recall + bigram cụm ("machine learning") |
| min_df / max_df | `1` / `0.9` | corpus nhỏ giữ từ hiếm; loại token quá phổ biến |
| stop_words | `None` (giữ) | giữ từ để khớp tiêu đề ngắn ("with", "for") |
| sublinear_tf / norm | `True` / `l2` | giảm ảnh hưởng tf cao, cosine chuẩn |

### 4.3 Semantic (FastText chính, Word2Vec baseline)
| Tham số | Giá trị | Lý do |
|---|---|---|
| vector_size | 200 | đủ biểu diễn, nhẹ cho corpus nhỏ |
| window | 5 | ngữ cảnh vừa phải |
| min_count | 2 | giữ nhiều vocab trên corpus nhỏ |
| epochs / sg | 30 / 1 | skip-gram tốt cho từ hiếm |
| workers | 1 | tất định (NFR-02) |
| FastText min_n/max_n | 3 / 6 | subword n-gram xử lý OOV/lỗi gõ |

Tokenizer: **T1** raw lowercase cho lexical; **T2** (bỏ stopword EN) cho semantic.

### 4.4 Hybrid
`hybrid = alpha * minmax(lexical) + (1-alpha) * minmax(semantic)` trên union candidate top-100.
Min-max chuẩn hóa **trong cùng lượt search**. Alpha chọn theo **mean Recall@5 trên dev**.

---

## 5. Kết quả chính (test split)

Recall@5 theo method × group (xem `reports/eval_results.csv` cho đầy đủ @1/@3/@5 + MRR):

| method | G1 | G2 | G3 | ALL |
|---|---|---|---|---|
| lexical | 0.80 | 0.40 | 0.20 | 0.4667 |
| semantic (FastText) | 0.80 | 0.40 | **0.60** | **0.6000** |
| semantic (Word2Vec) | 0.80 | 0.40 | 0.60 | 0.6000 |
| hybrid α=0.25 | 0.80 | 0.40 | 0.60 | 0.6000 |
| hybrid α=0.50 | 0.80 | 0.40 | 0.40 | 0.5333 |
| hybrid α=0.75 | 0.80 | 0.40 | 0.20 | 0.4667 |

**Alpha chọn trên dev** (mean Recall@5): α=0.25 → 0.600, **α=0.50 → 0.733**, α=0.75 → 0.600 ⇒ chọn **α\*=0.50**.

> Nhận xét chính: semantic/FastText vượt lexical rõ ở G3 (variation-robust, nhờ subword xử lý lỗi gõ),
> lexical mạnh ở G1 (khớp mã/tên chính xác). G2 (paraphrase) khó cho cả hai.

---

## 6. Hạn chế đã biết

1. **G2 (semantic paraphrase) chỉ đạt 0.40** — corpus nhỏ (1.563 doc) và mean-pooling pha loãng tín hiệu
   dài; FastText học trên chính corpus nên không có tri thức ngoài.
2. **Query tiếng Việt kém** — corpus tiếng Anh; query như "học máy cơ bản" gần như OOV hoàn toàn.
3. **G1 đôi lúc miss** khi nhiều doc cùng chứa cụm từ (vd "Molecular Biology" có 3 khóa cạnh tranh),
   ranking theo TF-IDF chưa ưu tiên khóa "chuẩn".
4. **Alpha chọn trên dev (α=0.5) không tối ưu trên test** (α=0.25/0.6 tốt hơn ở G3) → dev nhỏ (15 case),
   overfit nhẹ. Không tune lại test (NFR-07).
5. **MRR/Precision** thấp do nhiều query chỉ có 1 relevant doc.

---

## 7. Truy vết yêu cầu

Xem `reports/quality_report.md` (8 phần) để biết mapping FR ↔ AC ↔ kết quả.

# Hệ thống gợi ý phim cá nhân hóa (Đồ án 4)

Xây dựng hệ thống gợi ý phim cá nhân hóa bằng kỹ thuật học máy trên bộ dữ liệu MovieLens 1M.

## Cài đặt (Windows, PowerShell)

```powershell
cd D:\DoAn4-MovieRec
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py      # tải MovieLens 1M (~6 MB) vào data/raw/ml-1m
python scripts/fetch_tmdb.py         # (tùy chọn) poster + mô tả phim từ TMDB, cần TMDB_API_KEY trong file .env
```

## Chạy

```powershell
python -m pytest -q                  # kiểm thử trên dữ liệu giả, không cần tải dataset
python scripts/run_baselines.py      # so sánh nhanh các mô hình mốc -> results/baselines.csv
python scripts/tune.py               # chỉnh siêu tham số trên val -> results/best_params.json (vài phút)
python scripts/run_final.py          # huấn luyện lại trên train+val, chấm trên test -> results/final.csv
python scripts/run_two_stage.py      # kiến trúc hai tầng: truy hồi + LightGBM -> results/two_stage.csv
python scripts/run_ncf.py            # NeuMF trên GPU (cần PyTorch) -> results/ncf.csv
python scripts/build_semantic.py     # (tùy chọn) chỉ mục tìm phim bằng câu tiếng Việt, cần sentence-transformers
streamlit run app/streamlit_app.py   # mở web demo tại http://localhost:8501
```

## Cấu trúc

```
src/movierec/
  data.py        Nạp ML-1M, đánh lại chỉ số, chia train/val/test theo thời gian
  metrics.py     RMSE, Precision@K, Recall@K, NDCG@K, HitRate@K, Coverage
  evaluate.py    Quy trình đánh giá dùng chung (top-K, loại phim đã xem)
  models/
    base.py      Giao diện chung: fit(), score_users(), predict()
    baselines.py Popularity, BiasBaseline (mu + b_u + b_i)
    item_knn.py  Lọc cộng tác dựa trên phim (cosine + shrinkage)
    svd.py       Phân rã ma trận có độ lệch (Funk SVD), dự đoán điểm
    als.py       Phân rã ma trận cho phản hồi ngầm (implicit ALS), xếp hạng top-K
    content.py   Dựa trên nội dung: TF-IDF thể loại, từ khóa, diễn viên, mô tả (TMDB) + LSA
    hybrid.py    Lai: cộng có trọng số điểm đã chuẩn hóa z-score
    ncf.py       NeuMF (Neural Collaborative Filtering) bằng PyTorch
  ranker.py      Kiến trúc hai tầng: truy hồi ứng viên + xếp hạng lại bằng LightGBM
  semantic.py    Tìm phim theo ngữ nghĩa bằng mô hình nhúng đa ngôn ngữ
  service.py     Gói mô hình cho web; gợi ý cho người dùng mới bằng fold-in, kèm lý do
  db.py          SQLite: tài khoản (mật khẩu băm PBKDF2) và đánh giá
  tmdb.py        Ghép phim MovieLens với TMDB theo tên + năm
app/
  streamlit_app.py  Web: đăng ký/đăng nhập, hỏi sở thích, gợi ý, tìm phim, lịch sử
scripts/         download_data.py, run_baselines.py, tune.py, run_final.py, build_service.py
tests/           Kiểm thử tự động (pytest)
results/         Bảng kết quả
```

## Quy trình đánh giá

- Chia dữ liệu theo thời gian cho từng người dùng: 80% lượt chấm đầu để train, 10% val, 10% cuối test.
- Phim "liên quan" là phim được chấm >= 4 sao trong tập đánh giá.
- Mỗi mô hình gợi ý top-10 phim chưa xem; đo Precision, Recall, NDCG, HitRate, Coverage.
- Mô hình dự đoán điểm được đo thêm RMSE.

## Kết quả (tập test, MovieLens 1M)

| Mô hình | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | RMSE | vs Popularity |
|---|---|---|---|---|---|---|---|
| ContentBased | 0.0062 | 0.0154 | 0.0109 | 0.0549 | 0.5845 |  | -81.0% |
| BiasBaseline | 0.0177 | 0.0191 | 0.0214 | 0.1294 | 0.0189 | 0.9237 | -62.7% |
| SVD | 0.0213 | 0.0258 | 0.0266 | 0.1526 | 0.1711 | 0.8653 | -53.7% |
| Popularity | 0.0438 | 0.0505 | 0.0574 | 0.2625 | 0.0391 |  | +0.0% |
| ItemKNN | 0.0493 | 0.0790 | 0.0760 | 0.3256 | 0.1333 |  | +32.4% |
| NeuMF | 0.0563 | 0.0883 | 0.0838 | 0.3548 | 0.4212 |  | +46.0% |
| Hybrid | 0.0598 | 0.0955 | 0.0928 | 0.3677 | 0.1557 |  | +61.7% |
| ImplicitALS | 0.0611 | 0.0986 | 0.0935 | 0.3802 | 0.2245 |  | +62.9% |
| TwoStage | 0.0635 | 0.1067 | 0.0982 | 0.4005 | 0.2909 |  | +71.1% |

Tạo lại bảng này bằng `python scripts/summarize.py`.

## Lộ trình

- [x] Lớp 1a: dữ liệu, khung đánh giá, mô hình mốc (Popularity, BiasBaseline, ItemKNN)
- [x] Lớp 1b: SVD, implicit ALS, content-based, hybrid, chỉnh siêu tham số
- [x] Lớp 1c: web Streamlit, tài khoản, SQLite, cold-start, giải thích gợi ý
- [x] Lớp 2a: dữ liệu TMDB (poster, mô tả, từ khóa, diễn viên), phim tương tự theo nội dung
- [x] Lớp 2b: kiến trúc hai tầng (truy hồi + LightGBM), NeuMF
- [x] Lớp 2c: web dùng kiến trúc hai tầng cho người đã thích từ 5 phim, công thức trộn cho người mới
- [x] Lớp 3a: tìm phim bằng câu tiếng Việt (nhúng văn bản đa ngôn ngữ)
- [ ] Lớp 3b: trợ lý hội thoại, SASRec/LightGCN

## Nguồn dữ liệu

- MovieLens 1M (GroupLens Research).
- Poster và mô tả phim: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.

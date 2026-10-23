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
python scripts/run_lightgcn.py       # LightGCN, mạng nơ-ron đồ thị (cần PyTorch) -> results/lightgcn.csv
python scripts/run_sasrec.py         # SASRec, gợi ý theo chuỗi (cần PyTorch) -> results/sasrec.csv
python scripts/sasrec_order_check.py # SASRec khi xáo trộn thứ tự lịch sử -> results/sasrec_order.csv
python scripts/summarize.py          # gộp mọi kết quả -> results/summary.csv, summary.md
streamlit run app/streamlit_app.py   # mở web demo tại http://localhost:8501
```

Tìm phim theo mô tả và trợ lý hội thoại (tùy chọn):

```powershell
pip install -r requirements-optional.txt   # PyTorch, sentence-transformers
python scripts/eval_semantic.py            # so sánh 3 mô hình nhúng, lưu artifacts/semantic.npz
winget install Ollama.Ollama               # rồi mở lại PowerShell
ollama pull qwen3:4b
python scripts/eval_assistant.py           # đánh giá khả năng hiểu câu của trợ lý -> results/assistant_eval.csv
python scripts/build_blurbs.py             # viết sẵn câu giới thiệu tiếng Việt cho mọi phim (~7 giờ CPU, chạy tiếp được)
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
    lightgcn.py  LightGCN: lọc cộng tác bằng mạng nơ-ron đồ thị (PyTorch)
    sasrec.py    SASRec: gợi ý theo chuỗi bằng Transformer (PyTorch)
  ranker.py      Kiến trúc hai tầng: truy hồi ứng viên + xếp hạng lại bằng LightGBM
  semantic.py    Tìm phim theo ngữ nghĩa bằng mô hình nhúng đa ngôn ngữ (BGE-M3)
  beyond.py      Độ đa dạng, độ mới, đuôi dài; xếp hạng lại MMR
  assistant.py   Trợ lý hội thoại: LLM cục bộ (Ollama) hiểu câu, các mô hình gợi ý chọn phim
  service.py     Gói mô hình cho web; gợi ý cho người dùng mới bằng fold-in, kèm lý do
  db.py          SQLite: tài khoản (mật khẩu băm PBKDF2) và đánh giá
  tmdb.py        Ghép phim MovieLens với TMDB theo tên + năm
app/
  streamlit_app.py  Web: đăng ký/đăng nhập, hỏi sở thích, gợi ý, tìm phim, tìm theo mô tả, trợ lý, lịch sử
scripts/         Tải dữ liệu, chỉnh siêu tham số, huấn luyện, đánh giá; gpu_check.py, gpu_monitor.ps1 kiểm tra GPU
data/eval/       40 câu tìm theo mô tả và 24 câu hội thoại có nhãn, dùng để đánh giá
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
| NeuMF | 0.0564 | 0.0885 | 0.0839 | 0.3550 | 0.4217 |  | +46.2% |
| LightGCN | 0.0586 | 0.0878 | 0.0857 | 0.3567 | 0.3492 |  | +49.3% |
| Hybrid | 0.0598 | 0.0955 | 0.0928 | 0.3677 | 0.1557 |  | +61.7% |
| ImplicitALS | 0.0611 | 0.0986 | 0.0935 | 0.3802 | 0.2245 |  | +62.9% |
| TwoStage | 0.0635 | 0.1067 | 0.0982 | 0.4005 | 0.2909 |  | +71.1% |
| SASRec | 0.0999 | 0.2004 | 0.1779 | 0.5445 | 0.6603 |  | +209.9% |

SASRec cao nhất nhưng phụ thuộc mạnh vào thứ tự chấm điểm: khi xáo trộn thứ tự lịch sử, NDCG@10 giảm từ 0.1769 xuống
khoảng 0.0626 (`results/sasrec_order.csv`); huấn luyện lại ngay từ đầu trên chuỗi xáo trộn cũng chỉ đạt 0.0663
(`results/sasrec_shuffled.csv`). Web vì vậy dùng kiến trúc hai tầng (TwoStage).

Tìm phim theo mô tả tiếng Việt (BGE-M3 + độ phổ biến, 20 câu kiểm tra): Hit@1 0.75, Hit@5 0.90, MRR 0.823.
Trợ lý hội thoại (qwen3:4b trên CPU, 24 câu): đúng loại yêu cầu 0.875, F1 thể loại 0.875, khoảng 7.6 giây mỗi câu.
Câu giới thiệu tiếng Việt viết sẵn cho 3.639/3.673 phim có mô tả (99,1%): mỗi lượt chat trên web còn 12-17 giây (trước đó 42-50 giây).

Ngoài độ chính xác (`python scripts/beyond_accuracy.py` -> `results/beyond.csv`, `mmr.csv`, `pop_penalty.csv`):
thiên lệch độ phổ biến rất rõ, TwoStage chỉ dành 0,8% vị trí gợi ý cho 2.558 phim đuôi dài (69% kho phim), Gini 0,935.
Xếp hạng lại bằng MMR theo thể loại với lambda 0,8: NDCG@10 giảm 0,6%, độ đa dạng thể loại (ILD) tăng 6%; web có lựa chọn
"Độ đa dạng thể loại" với mặc định là mức này. Phạt độ phổ biến (beta 0,3) tăng độ phủ từ 29,1% lên 33,0%, NDCG@10 giảm 2,4%.

Giải thích bộ xếp hạng bằng SHAP (`python scripts/explain_ranker.py` -> `results/shap_groups.csv`): với 10 phim được gợi ý,
Implicit ALS là lý do đẩy lên mạnh nhất ở 99,5% số phim (trung bình |SHAP| 0,465, gấp 4 lần ItemKNN); độ quan trọng theo gain
đánh giá quá cao các đặc trưng phụ như năm phát hành. Trên web, mỗi phim của kiến trúc hai tầng có mục "Vì sao gợi ý phim này?".

Tạo lại bảng này bằng `python scripts/summarize.py`.

## Lộ trình

- [x] Lớp 1a: dữ liệu, khung đánh giá, mô hình mốc (Popularity, BiasBaseline, ItemKNN)
- [x] Lớp 1b: SVD, implicit ALS, content-based, hybrid, chỉnh siêu tham số
- [x] Lớp 1c: web Streamlit, tài khoản, SQLite, cold-start, giải thích gợi ý
- [x] Lớp 2a: dữ liệu TMDB (poster, mô tả, từ khóa, diễn viên), phim tương tự theo nội dung
- [x] Lớp 2b: kiến trúc hai tầng (truy hồi + LightGBM), NeuMF
- [x] Lớp 2c: web dùng kiến trúc hai tầng cho người đã thích từ 5 phim, công thức trộn cho người mới
- [x] Lớp 3a: tìm phim bằng câu tiếng Việt (nhúng văn bản đa ngôn ngữ)
- [x] Lớp 3b: trợ lý hội thoại tiếng Việt (Ollama, qwen3:4b), SASRec và phép thử xáo trộn thứ tự
- [x] Thí nghiệm bổ sung: SASRec huấn luyện trên chuỗi xáo trộn
- [x] LightGCN: NDCG@10 0.0857, xấp xỉ NeuMF, chưa vượt Implicit ALS
- [x] Câu giới thiệu tiếng Việt viết sẵn cho mọi phim, trợ lý nhanh hơn khoảng 3 lần
- [x] Độ đa dạng, độ mới, thiên lệch độ phổ biến; xếp hạng lại MMR và lựa chọn mức đa dạng trên web
- [x] Giải thích từng gợi ý bằng giá trị SHAP của bộ xếp hạng LightGBM

## Nguồn dữ liệu

- MovieLens 1M (GroupLens Research).
- Poster và mô tả phim: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.

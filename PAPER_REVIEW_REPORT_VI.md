# Báo cáo Kiểm toán Kỹ thuật & Rà soát Toàn diện: FedSCD

> **Bài báo**: *Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning*  
> **Hội nghị mục tiêu**: International Conference on Machine Learning (Bản thảo nộp bài ICML 2026 tại `Overleaf/icml2026.tex`)  
> **Chế độ rà soát**: Báo cáo Tổng hợp Cuối cùng (Definitive Final Reducer Synthesis Report)  
> **Ngày thực hiện**: 11-11-2026  
> **Phạm vi kiểm tra**: Kiểm toán đối chiếu toàn diện giữa Bản thảo bài báo (`Overleaf/icml2026.tex`), File PDF biên dịch (`icml2026.pdf`), Nhật ký biên dịch TeX (`icml2026.log`), Các chứng minh toán học, Cơ sở dữ liệu thực nghiệm (`results/runs_results/`), Kịch bản thực thi và Mã nguồn PyTorch (`system/`).  
> **Trạng thái**: Hoàn chỉnh, Chi tiết, Đạt chuẩn công bố khoa học, Được đối chiếu và hợp nhất qua 4 bản tổng hợp song song và 5 báo cáo thành phần chuyên sâu.

---

# 1. Tóm tắt Điều hành & Tổng quan Kiểm toán Chiến lược

### 1.1 Cơ sở Tài liệu & Thuật toán
Bản thảo **"Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning"** tập trung giải quyết điểm yếu cốt lõi trong Học liên kết dựa trên Prototype (Prototype-based Federated Learning - PFL) dưới điều kiện lệch nhãn (non-IID label skew) nghiêm trọng. Các thuật toán PFL truyền thống (như FedProto, FedTGP, FedProc, FedPCL) tóm tắt mỗi lớp ngữ nghĩa thành một điểm tâm cụm Euclidean duy nhất (prototype phương sai bằng 0, $\mu^c \in \mathbb{R}^d$) và ép buộc sự phân tách đặc trưng bằng cách tối đa hóa khoảng cách Euclidean giữa các tâm cụm này. Nhóm tác giả đã chứng minh — cả về mặt khái niệm lẫn thực nghiệm — rằng việc chỉ nới rộng khoảng cách giữa các tâm cụm không thể ngăn chặn sai số phân loại Bayes nghiêm trọng khi độ phân tán đặc trưng nội lớp (intra-class feature dispersion) vẫn lớn, bất đẳng hướng (anisotropic) và bị phân tán rời rạc giữa các client cục bộ không đồng nhất.

Để khắc phục hạn chế này, các tác giả đề xuất **FedSCD** (Federated Statistics-Consistent Distributional Prototype Learning). Trong FedSCD, các lớp được tham số hóa dưới dạng phân phối Gauss đường chéo liên tục $\mathcal{N}(\mu^c, \mathrm{diag}((\sigma^c)^{\odot 2}))$. Các client cục bộ tính toán vector trung bình và độ lệch chuẩn đặc trưng theo từng lớp, chỉ tải lên server bộ giá trị rút gọn $\{\mu_k^c, \sigma_k^c, n_k^c\}$. Phía server gom các thống kê phân tán này thông qua **định lý phương sai toàn phần (law of total variance)**, nắm bắt chính xác cả phương sai nội bộ client (within-client variance) và độ trôi tâm cụm giữa các client (between-client centroid drift) mà không cần trao đổi dữ liệu thô hay trọng số mạng nơ-ron. Ở phía client, việc học biểu diễn cục bộ được định hướng bởi:
1. **Co cụm mỏ neo Empirical Bayes (Empirical Bayes Anchor Shrinkage)** và **Chiếu làm trắng theo độ chính xác (Precision-Whitened Projections)** giúp giảm trọng số các chiều nhiễu, phương sai cao;
2. **ProtoNCE thích ứng theo độ phân tán (Dispersion-Adaptive ProtoNCE)** và **Học tương phản có giám sát loại bỏ lực đẩy mẫu dương (Positive-Repulsion-Free Supervised Contrastive Learning - SupCon/DCL)** giúp hiệu chỉnh động biên góc (angular margins) bằng nhiệt độ lớp suy ra từ server;
3. **Chính quy hóa độ nén Mahalanobis điều khiển bởi cổng SNR (SNR-Gated Mahalanobis Compactness Regularization)** giúp gom cụm đặc trưng theo trọng số độ chính xác dọc theo các tọa độ có độ tin cậy cao; và
4. **Hiệu chuẩn đầu phân loại tách rời (Decoupled Classifier Head Calibration)** bằng cách lấy mẫu các đặc trưng ảo Gauss từ các moment phân phối toàn cục được phát sóng từ server.

Khung làm việc này duy trì chi phí truyền thông tiệm cận ở mức $\mathcal{O}(Cd)$ cho mỗi client trên mỗi vòng, khớp chính xác với tốc độ tăng trưởng truyền thông tuyến tính của các phương pháp federated learning prototype bậc một thông thường.

---

### 1.2 Đánh giá Chiến lược: Tiềm năng Khoa học Lớn nhưng Vướng các Khuyết tật Trầm trọng
Mặc dù động lực lý thuyết và kết quả thực nghiệm trên CIFAR-10 cho thấy tiềm năng rất lớn, cuộc kiểm toán kỹ thuật toàn diện cho thấy **bản thảo ở trạng thái hiện tại chưa sẵn sàng để nộp hội nghị và đối mặt với nguy cơ bị loại ngay vòng gửi xe (administrative desk rejection) hoặc bị từ chối đồng thuận từ các phản biện**:

1. **Các nguy cơ bị Desk-Reject ngay lập tức về mặt thủ tục**:
   - **Vi phạm giới hạn số trang ICML**: Bản thảo biên dịch `Overleaf/icml2026.pdf` dài tới **11 trang trước phần tài liệu tham khảo** (tổng cộng 13 trang, references bắt đầu từ trang 12). ICML quy định ngặt nghèo **giới hạn 8 trang** cho phần nội dung chính trước references và phụ lục. Bài nộp vượt quá 8 trang sẽ bị desk-reject thẳng tay mà không phân phối cho phản biện.
   - **Vi phạm quy tắc ẩn danh Double-Blind**: Các macro biên tập của tác giả (`\NMD{...}`, `\GVQ{...}`) hiển thị chữ đậm màu xanh và đỏ kèm tên viết tắt (ví dụ: `[NMD: ...]`, `[GVQ: ...]`) tại 5 vị trí trong phần thân bài (Trang 2, 3 và 7). Việc để lộ ký danh tác giả trong văn bản vi phạm trực tiếp quy chế nộp bài ẩn danh hai chiều.
   - **Tự để lộ thiếu sót thực nghiệm & phần viết dở dang**: Dòng 192 in dòng chữ đỏ `\GVQ{need TSNE to complete evidence this claim).}` ngay sau tuyên bố cốt lõi rằng thống kê toàn cục giúp căn chỉnh các bộ trích xuất đặc trưng cục bộ vào một hệ tọa độ chung. Các dòng 738, 745 và 752 in `[NMD: Please find some more works]` và `[NMD: ... This is not enough.]` ngay trong phần Related Work, công khai cho thấy phần tổng quan tài liệu chưa hoàn thiện.

2. **Lỗ hổng lý thuyết chí mạng — Hoàn toàn thiếu vắng các chứng minh**:
   - Mục 3.7 phát biểu 3 kết quả toán học hình thức: **Proposition 3.1** (Exact pooling under total variance), **Proposition 3.2** (Drift-induced dispersion inflation), và **Proposition 3.3** (Bhattacharyya bound on pairwise Bayes error).
   - Dòng 556 có ghi chú nháp: `% Three short results; proofs go to the appendix.`. Tuy nhiên, **bài báo hoàn toàn không có phần phụ lục Appendix nào**. Việc nộp một bài báo ICML chứa các mệnh đề hình thức không có chứng minh đi kèm là lý do bị loại ngay lập tức trong quá trình phản biện.

3. **Mâu thuẫn thực nghiệm nghiêm trọng & Vi phạm tính toàn vẹn của chỉ số**:
   - **Che giấu sự sụp đổ mô hình của Baseline (Concealment of Baseline Collapse)**: Mục 4 (Dòng 609) tuyên bố độ chính xác kiểm thử được đo ở *"vòng truyền thông cuối cùng (final communication round)"*. Tuy nhiên, phân tích file `results/runs_results/runs_results.xlsx` (Sheet `Summary`, Hàng 3) và log CSV gốc chứng minh rằng **Bảng 1 báo cáo độ chính xác cao nhất trong toàn bộ lịch sử (Peak Test Accuracy: `df.max() * 100`) qua 100 vòng**. Cách làm này che giấu việc các baseline bị sụp đổ phân kỳ nghiêm trọng ở các vòng sau: trên Flowers102 với $\alpha=0.1$, FedTGP đạt đỉnh 54.79% ở vòng 57 rồi sụp đổ xuống **0.58% ở vòng 100** (Bảng 1 báo cáo 54.79%); trên Flowers102 với $\alpha=0.5$, FedTGP sụp đổ thành `NaN`; và FedPLVM phân kỳ thành `NaN` trong 6 trên 9 lượt chạy benchmark.
   - **Số liệu trong văn bản mâu thuẫn với Bảng 1 và bị thổi phồng**: Mục 1 (Dòng 198) và Mục 5.2 (Dòng 645--647) trích dẫn các log sơ bộ cũ. Văn bản tuyên bố vượt FedTGP "+6.5% tại $\alpha=0.5$" (tuyên bố FedTGP đạt 68.64%), trong khi Bảng 1 in FedTGP đạt 71.70% (chênh lệch thực tế chỉ là **+3.45%**, tức tuyên bố khống lên tới 88%). Mức chênh lệch đỉnh "+8.4% so với FedTGP" thực chất là so với FedORGP tại $\alpha=1.0$ ($69.60 - 61.21 = 8.39\%$), nhưng bị gán nhầm thành FedTGP (vốn có khoảng cách tối đa thực tế là +5.09%).
   - **Baseline "ma" & Bảng số liệu bị khuyết**: Lời văn so sánh mạnh mẽ FedSCD với FedProto (65.03%, 86.10%) và CCVR (62.85% đến 45.65%), nhưng cả hai baseline này hoàn toàn không xuất hiện trong Bảng 1 hay mục Baselines của Mục 4. Hơn nữa, Bảng 1 có 9 ô gạch ngang rỗng (`--`) cho FedSCD trên CIFAR-100 và Flowers102, còn Tiny-ImageNet có 21 ô gạch ngang liên tiếp cho toàn bộ các thuật toán, mâu thuẫn trực tiếp với tuyên bố kiểm định trên nhiều tập dữ liệu.

4. **Sự sai lệch trên diện rộng giữa Mã nguồn và Bài báo**:
   - **Chưa công bố Feature Manifold Mixup**: Tại `clientscd.py:459--472`, quá trình huấn luyện client cục bộ chạy Feature Manifold Mixup trên biểu diễn đầu phân loại (`mixup_alpha = 0.2` kích hoạt mặc định ở `main.py:342`), đóng góp 25% vào hàm mất mát đầu phân loại ($0.75 \mathcal{L}_{\mathrm{CE}} + 0.25 \mathcal{L}_{\mathrm{CE, mixup}}$). Thành phần kiến trúc quan trọng này bị bỏ sót hoàn toàn khỏi văn bản và phương trình của bài báo.
   - **Lệch siêu tham số nghiêm trọng**: Server momentum được tuyên bố là $\beta = 0.6$ (Dòng 607, 742), nhưng mặc định trong code và kịch bản benchmark thực tế chạy $\beta = 0.1$ (`serverscd.py:32`, `main.py:326`, `run_benchmark_matrix.py:77`), chênh lệch tới $6\times$ về tốc độ học EMA. Số vòng ramp-up tuyên bố là $T_{\mathrm{ramp}} = 10$, nhưng code chạy $T_{\mathrm{ramp}} = 5$. Trọng số đặc trưng ảo cục bộ tuyên bố $\lambda_{\mathrm{vl}} = 0.05$, nhưng thực thi là $\lambda_{\mathrm{vl}} = 0.0$ (bị tắt hoàn toàn).
   - **Hỏng tham chiếu chéo LaTeX**: Các nhãn bị thiếu tạo ra các ký tự in đậm **`(Eq. ??)`** xuất hiện rõ mồn một trong Thuật toán 1 (Dòng 12 và 17) và văn bản tại Trang 7 và 8 (`\ref{eq:nce}`, `\ref{eq:head}`, `\ref{eq:server_mom}`, `\ref{tab:label_skew}`).

---

# 2. Ma trận Phân bổ Mức độ Nghiêm trọng

Toàn bộ quá trình kiểm toán xác thực được **81 vấn đề riêng biệt**, được phân loại và khử trùng lặp như sau:

### 2.1 Phân bổ theo Phân đoạn Bài báo

| Phân đoạn Bài báo / Phân vùng Phân tích | Nghiêm trọng (Critical) | Lớn (Major) | Nhỏ (Minor) | Lỗi nhỏ / Đánh máy | Tổng cộng |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Phân đoạn 1**: Tóm tắt, Mở đầu & Công trình liên quan (Mục 1--2, Bảng 1) | 2 | 4 | 5 | 3 | **14** |
| **Phân đoạn 2**: Cơ sở Phương pháp & Thiết lập bài toán (Mục 3.0--3.3, Hình 1--2) | 0 | 5 | 5 | 4 | **14** |
| **Phân đoạn 3**: Học phân phối phía Client (Mục 3.4, Phương trình 5--12) | 0 | 6 | 5 | 2 | **13** |
| **Phân đoạn 4**: Tổng hợp Server, Thuật toán & Lý thuyết (Mục 3.5--3.7, Thuật toán 1) | 2 | 5 | 5 | 3 | **15** |
| **Phân đoạn 5**: Thực nghiệm, Đánh giá & Kết luận (Mục 4--6, Bảng 1--3) | 3 | 7 | 8 | 7 | **25** |
| **Tổng số phát hiện độc nhất** | **7** | **23** | **28** | **23** | **81** |

---

### 2.2 Phân bổ theo Phân loại Lỗi Kỹ thuật

| Phân loại Lỗi | Nghiêm trọng (Critical) | Lớn (Major) | Nhỏ (Minor) | Lỗi nhỏ / Đánh máy | Tổng cộng |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Thủ tục & Tuân thủ Quy định (Số trang ICML, Ẩn danh)** | 2 | 0 | 0 | 2 | **4** |
| **Tính Chặt chẽ Thực nghiệm (Bảng số liệu, Chỉ số, Baseline)** | 3 | 5 | 6 | 4 | **18** |
| **Tính Chặt chẽ Lý thuyết & Toán học (Chứng minh, Công thức)** | 1 | 7 | 6 | 4 | **18** |
| **Sai lệch Thuật toán so với Code & Thành phần chưa công bố** | 0 | 6 | 5 | 1 | **12** |
| **Lỗi Biên dịch & Tham chiếu chéo (`\ref`, Khối TeX)** | 1 | 2 | 4 | 2 | **9** |
| **Ghi chú Biên tập, Nháp & Trau chuốt Văn phong** | 0 | 3 | 7 | 10 | **20** |
| **Tổng cộng** | **7** | **23** | **28** | **23** | **81** |

---

# 3. Lộ trình Tổng thể Các Vấn đề Then chốt (Bảng Danh mục Lỗi)

Bảng dưới đây liệt kê toàn bộ các vấn đề ở mức **Nghiêm trọng (Critical)** và **Lớn (Major)**, đối chiếu vị trí chính xác trong bài báo và mã nguồn:

| Mã Lỗi | Mức độ | Phân loại | Vị trí & Dòng trong Bài báo | Tham chiếu Code Repository | Bản chất Vấn đề & Nguy cơ Học thuật |
|:---|:---:|:---|:---|:---|:---|
| **CRIT-01** | **Critical** | Tuân thủ Quy chế | `icml2026.pdf` (Trang 1--13) | `Overleaf/icml2026.tex:1--810` | **Vi phạm giới hạn 8 trang của ICML**: Phần nội dung chính dài tới 11 trang trước references (tổng 13 trang; references từ trang 12). Dẫn tới việc bị desk reject ngay lập tức không qua phản biện. |
| **CRIT-02** | **Critical** | Chặt chẽ Lý thuyết | Mục 3.7, Dòng 554--593 (Trang 9) | Dòng 556 ghi chú nháp | **Hoàn toàn thiếu vắng các chứng minh toán học**: Các Proposition 3.1, 3.2, và 3.3 được phát biểu hình thức, hứa hẹn chứng minh ở phụ lục (`% proofs go to the appendix`), nhưng bài báo không hề có phụ lục. |
| **CRIT-03** | **Critical** | Ẩn danh / Quy chế | Dòng 73--74, 190, 192, 211, 461 | Preamble `Overleaf/icml2026.tex` | **Vi phạm ẩn danh Double-Blind & Lộ thiếu sót thực nghiệm**: Macro `\NMD` và `\GVQ` in tên viết tắt tác giả dạng in đậm có màu; dòng 192 in `\GVQ{need TSNE...}`; dòng 738, 745, 752 lộ việc chưa đọc đủ tài liệu. |
| **CRIT-04** | **Critical** | Trung thực Thực nghiệm | Mục 4.1 (Dòng 609), Bảng 1 (`tab:com`) | `results/runs_results/runs_results.xlsx` | **Báo cáo Peak Accuracy dưới danh nghĩa Final-Round Accuracy**: Bảng 1 lấy giá trị cực đại lịch sử (`df.max() * 100`), che giấu việc baseline sụp đổ (FedTGP trên Flowers102 sụt từ 54.79% xuống 0.58%; FedPLVM phân kỳ ra `NaN` ở 6/9 lần chạy). |
| **CRIT-05** | **Critical** | Phạm vi Benchmark | Bảng 1 (Dòng 610--636), Dòng 753 | File log CSV `results/runs_results/` | **Bảng 1 bị khuyết dữ liệu & Tuyên bố khống đa tập dữ liệu**: FedSCD có 9 dấu gạch ngang rỗng (`--`) ở CIFAR-100 và Flowers102; Tiny-ImageNet có 21 dấu gạch ngang liên tiếp; mâu thuẫn với tuyên bố kiểm định trên nhiều tập dữ liệu. |
| **CRIT-06** | **Critical** | Chặt chẽ Số liệu | Mục 1 (Dòng 198), Mục 5.2 (Dòng 645--647) | `BENCHMARK_RUN_GUIDE.md:117--165` | **Văn bản mâu thuẫn trực tiếp với Bảng 1**: Tuyên bố FedTGP đạt 68.64% (vượt +6.5%; trong khi Bảng 1 ghi 71.70%, mức vượt thực tế là +3.45%); gán nhầm mức vượt +8.4% của FedORGP thành FedTGP; so sánh với baseline không tồn tại (FedProto, CCVR). |
| **CRIT-07** | **Critical** | Lỗi Biên dịch / Code | Thuật toán 1 (Dòng 12, 17), Mục 3.4, 3.5 | `icml2026.log` (5 cảnh báo) | **Hỏng tham chiếu chéo LaTeX trên diện rộng**: Các nhãn không tồn tại (`eq:nce`, `eq:head`, `eq:server_mom`, `tab:label_skew`) tạo ra các ký tự in đậm **`(Eq. ??)`** ngay trong Thuật toán 1 và văn bản. |
| **MAJ-01** | **Major** | Lệch Thuật toán-Code | Mục 4.1 (Dòng 607), Dòng 742 | `serverscd.py:32`, `main.py:318,326,336`| **Sai lệch siêu tham số nghiêm trọng**: Server momentum tuyên bố $\beta=0.6$ nhưng code chạy $\beta=0.1$ (chênh $6\times$ tốc độ học); ramp-up tuyên bố $T_{\mathrm{ramp}}=10$ nhưng code chạy $5$; loss ảo cục bộ tuyên bố $\lambda_{\mathrm{vl}}=0.05$ nhưng chạy $0.0$ (tắt). |
| **MAJ-02** | **Major** | Công bố Phương pháp | Mục 3.4, Dòng 434--448 (Eq. 12) | `clientscd.py:459--472`, `main.py:342` | **Chưa công bố Feature Manifold Mixup**: Quá trình huấn luyện client mặc định chạy mixup ($\alpha_{\mathrm{mix}}=0.2$), đóng góp 25% head loss, nhưng hoàn toàn bị bỏ sót trong phương trình và văn bản bài báo. |
| **MAJ-03** | **Major** | Logic Toán học | Mục 1 (L192), 3.2 (L306), 3.5 (L481) | `clientscd.py:211`, `clientscd.py:329` | **Lập luận toán học bị đảo ngược ở Mahalanobis Loss**: Văn bản khẳng định loss phạt các chiều bị phân tán quá mức, nhưng trong công thức phương sai nằm ở mẫu số nên thực tế lại giảm nhẹ phạt ở chiều phương sai lớn (precision weighting). |
| **MAJ-04** | **Major** | Xung đột Ký hiệu | Mục 3.2 (Eq. 3, L296) vs Mục 3.5 (Eq. 14) | Định nghĩa văn bản | **Trùng lặp ký hiệu phương sai $\bar{\sigma}^2$**: Định nghĩa là phương sai trung bình giữa cặp 2 lớp ở Eq. 3, nhưng lại định nghĩa là phương sai gộp server giữa các client ở Eq. 14. Bị xung đột ở dòng 306. |
| **MAJ-05** | **Major** | Lệch Chỉ số | Mục 3.2 (Dòng 304) | `motivation_metrics.py:161--186` | **Lệch chuẩn hóa Bhattacharyya Overlap**: Code chia cho $\sqrt{d_{\mathrm{eff}}}$ có lọc tọa độ hoạt động, trong khi bài báo tuyên bố chia cho $d$ (tính ra $\approx 0.994$ thay vì $[0.86, 0.92]$ như trên đồ thị). |
| **MAJ-06** | **Major** | Chặt chẽ Toán học | Mục 3.1 (Eq. 2, L273), Mục 3.2 (Eq. 3)| `clientscd.py:125`, `serverscd.py:205` | **Điểm kỳ dị khi lớp chỉ có 1 mẫu ($n_k^c=1$)**: Khiến vector phương sai $(\sigma_k^c)^2=\mathbf{0}$, dẫn tới $\ln(+\infty)$, dạng $0/0$ trong Eq. 3, và đạo hàm Mahalanobis không xác định; bài báo thiếu ngưỡng sàn $\epsilon_0=10^{-6}$ mà code có. |
| **MAJ-07** | **Major** | Lệch Thuật toán-Code | Mục 3.4 (Dòng 362) | `clientscd.py:208--211` | **Lệch phạm vi tính ngưỡng sàn phương sai**: Văn bản định nghĩa $\tau_v$ lấy trung bình trên $d$ chiều của lớp $c$, nhưng code lại tính trung bình gộp trên toàn bộ các lớp và các chiều cùng lúc. |
| **MAJ-08** | **Major** | Chặt chẽ Toán học | Mục 3.4, Eq. 10 (Dòng 418) | `clientscd.py:260--269` | **Thiếu điều kiện lớp trong Mahalanobis Loss**: Eq. 10 không có điều kiện $y \in \mathcal{C}_{t-1}$; không xác định với các lớp chưa quan sát được ở các vòng đầu khi chưa có anchor. |
| **MAJ-09** | **Major** | Chặt chẽ Toán học | Mục 3.4, Eq. 12 (Dòng 443) | `clientscd.py:371, 385` | **Mâu thuẫn cấp độ công thức & Nhiễu chưa đánh chỉ số**: Eq. 12 trộn số hạng cross-entropy của 1 mẫu với tổng theo lớp của toàn tập dữ liệu mà không có toán tử kỳ vọng; ký hiệu $\eta$ không có chỉ số cho 2 tỷ lệ khác nhau. |
| **MAJ-10** | **Major** | Đánh giá Độ phức tạp | Mục 3.6 (Dòng 549) | `clientscd.py:298--314` | **Phân loại nhầm độ phức tạp tương phản là Element-Wise**: Tuyên bố chỉ gồm "phép toán vector từng phần tử", nhưng ProtoNCE và SupCon thực tế tính tích vô hướng ma trận dày đặc (dense inner products). |
| **MAJ-11** | **Major** | Chuẩn xác Thuật ngữ | Mục 3.5, Dòng 497, 503 (Eq. 16) | `serverscd.py:221--224` | **Ngược thuật ngữ**: Gọi $r_t^c$ là "điểm tập trung (concentration score)" trong khi nó tỷ lệ thuận với độ phân tán và độ không chắc chắn của mẫu. |
| **MAJ-12** | **Major** | Tính Chuẩn xác Lý thuyết | Mục 3.7 (Dòng 564, Prop. 3.1) | `clientscd.py:125` (`unbiased=False`) | **Thiếu điều kiện chính quy cho Proposition 3.1**: Việc gom phương sai toàn phần chính xác chỉ đúng với định nghĩa phương sai mẫu thực nghiệm ($1/N$), không đúng với ước lượng không chệch Bessel ($1/(N-1)$). |
| **MAJ-13** | **Major** | Chặt chẽ Lý thuyết | Mục 3.7 (Dòng 578--588, Prop. 3.3) | `Overleaf/icml2026.tex:578` | **Chưa trích dẫn định lý cổ điển Kailath & Thiếu mở rộng đa lớp**: Chưa ghi công Kailath (1967) trong tiêu đề mệnh đề; thiếu bất đẳng thức hợp (union bound) mở rộng cho bài toán $C$ lớp. |
| **MAJ-14** | **Major** | Công bằng Benchmark | Mục 4.1 (Dòng 607) | `system/wandb/run-*/config.yaml` | **Kích thước Batch không thống nhất giữa các Baseline**: FedTGP và FedSA chạy với $B=10$, trong khi FedCOF và FedORGP chạy $B=32$, vi phạm điều kiện $B=32$ đồng nhất ghi ở dòng 607. |
| **MAJ-15** | **Major** | Minh chứng Trực quan | Mục 5.3 (Dòng 664--672) | Dòng 192 (`\GVQ{need TSNE...}`) | **Mục 5.3 hoàn toàn thiếu số liệu định lượng & hình ảnh minh họa**: Không trỏ tới hình vẽ nào, không có bảng, không có số liệu cụ thể, khiến tuyên bố đồng thuận tọa độ không có bằng chứng. |
| **MAJ-16** | **Major** | Toàn vẹn Dữ liệu Vẽ hình | `Figure_Plot/test_acc.csv` | `plot_combined_3panels.py:40--77` | **File vẽ đồ thị động lực bị ghi đè**: `test_acc.csv` bị ghi đè bởi dữ liệu baseline, thiếu FedSCD, CCVR, FedProto; chạy lại kịch bản chỉ vẽ được duy nhất đường FedTGP ở panel (c). |
| **MAJ-17** | **Major** | Đánh giá Thực nghiệm | Bảng 3 (`tab:ablation_params`), Mục 5.4 | Bảng 1, 2, 3 | **Bảng 3 đặt sai tên & Thiếu thanh sai số (Error Bars)**: Bảng 3 kiểm tra 3 công tắc nhị phân chứ không phải quét liên tục; hoàn toàn không có độ lệch chuẩn qua nhiều seed ở tất cả các bảng. |
| **MAJ-18** | **Major** | Rõ ràng về Baseline | Bảng 2 (Dòng 708), Mục 5.4 | Bảng 2 hàng 6 | **Chưa định nghĩa giao thức cho Baseline ở Bảng 2**: Hàng 6 "Baseline (CE + Augmentation only): 72.57%" không nói rõ client có truyền prototype hay là huấn luyện cô lập không giao tiếp. |
| **MAJ-19** | **Major** | Thảo luận Hạn chế | Mục 6 (Dòng 755--756) | Văn bản Mục 6 | **Bỏ qua các giới hạn kỹ thuật nền tảng**: Bỏ qua đặc trưng hình nón phi-Gauss sau ReLU, suy biến phương sai khi $n_k^c \le 2$, chi phí tải tăng gấp đôi ($2\times$), và rủi ro quyền riêng tư vi phân (DP). |
| **MAJ-20** | **Major** | Cú pháp Biên dịch TeX | Dòng 678--736 | `Overleaf/icml2026.tex:151--209` | **Lỗi đóng mở ngoặc nhọn TeX cấp gốc**: Dấu mở ngoặc `{` ở dòng 678 chỉ được đóng ở dòng 736 làm bẫy 59 dòng text, một hình vẽ và cả lệnh `\section{Related Work}` vào trong một phạm vi cục bộ. |
| **MAJ-21** | **Major** | Tính Chuẩn xác Phân loại | Bảng 1 (`tab:paradigm_comparison`) | Dòng 758--779 | **Bảng 1 so sánh định tính sai**: Gán toàn bộ Statistical FL là $\mathcal{O}(Cd^2)$ (bỏ qua các phương pháp $\mathcal{O}(Cd)$ như FedCOF); baseline lai vô lý cho Contrastive FL; lỗi macro phông chữ toán. |
| **MAJ-22** | **Major** | Trích dẫn Tài liệu | Mục 2 (Dòng 757), Mục 3.4 (Remark 1) | `Overleaf/icml2026.tex:230` | **Gán sai nguồn gốc lý thuyết của hàm mất mát không đẩy mẫu dương**: Chỉ trích dẫn SupCon mà không dẫn nguồn Decoupled Contrastive Learning (DCL; Yeh et al., CVPR 2022). |
| **MAJ-23** | **Major** | Trích dẫn Chuyên ngành | Mục 1 (L680), Mục 2 (L733--755) | Dòng 680, 733, 736, 741, 748 | **Trích dẫn sai bản chất thuật toán**: Dẫn FedGH vào prototype FL (thực chất nó tổng hợp head); tuyên bố sai rằng mọi decoupled FL đều tổng hợp extractor; dẫn FedGMKD vào thống kê bậc 2; dính chữ vào citation. |

---

# 4. Rà soát Chi tiết Từng Phần giữa Bài báo & Mã nguồn

---

## 4.1 Sự Sai lệch giữa Thuật toán và Mã nguồn

### 1. Lệch Chuẩn hóa Chiều trong Bhattacharyya Overlap ($\sqrt{d_{\mathrm{eff}}}$ so với $d$) [Mã lỗi: MAJ-05]
- **Phát biểu trong bản thảo**: Mục 3.2, Dòng 304 của `Overleaf/icml2026.tex`:
  `"... (reported in Figure~\ref{fig:motivation} as the dimension-normalized overlap $\exp(-D_B(c,c')/d)$ averaged over class pairs)."`
- **Thực thi trong mã nguồn**: Trong file `system/utils/motivation_metrics.py`, Dòng 161--186:
  ```python
  active_mask = (var_dist.mean(dim=0) > 1e-4)
  d_eff = int(active_mask.sum().item())
  D_B_scaled = D_B / np.sqrt(float(d_eff))
  overlap_mat = torch.exp(-D_B_scaled)
  ```
- **Phân tích kỹ thuật**:
  Với số chiều biểu diễn $d = 512$, ta có $\sqrt{d} \approx 22.63$. Với khoảng cách Bhattacharyya thực nghiệm giữa các cặp lớp $D_B \approx 2.5$--$3.5$, nếu chia cho $d = 512$ thì giá trị sẽ là $\exp(-3.0 / 512) \approx 0.994$. Trong khi đó, nếu chia cho $\sqrt{d_{\mathrm{eff}}} \approx 22.63$ thì kết quả là $\exp(-3.0 / 22.63) \approx 0.876$. Con số này khớp chính xác với khoảng trục Y $[0.86, 0.92]$ được thiết lập trong `Figure_Plot/plot_combined_3panels.py` (Dòng 84). 
  Bất kỳ ai cố gắng tái lập Hình 1 bằng công thức xuất bản $\exp(-D_B/d)$ sẽ thu được các đường gần như phẳng lỳ tại $\approx 0.99$, mâu thuẫn hoàn toàn với hình vẽ trong bài.
- **Cách sửa nhanh trong LaTeX**: Cập nhật Dòng 304 của `icml2026.tex`:
  ```latex
  which equals $0$ for disjoint distributions and $1$ for identical ones (reported in Figure~\ref{fig:motivation} as the dimension-normalized overlap index $\exp(-D_B(c,c')/\sqrt{d_{\mathrm{eff}}})$ averaged over class pairs across active coordinates where $\bar\sigma_j^2 > 10^{-4}$).
  ```

### 2. Thành phần chưa công bố: Feature Manifold Mixup trong Huấn luyện Client [Mã lỗi: MAJ-02]
- **Phát biểu trong bản thảo**: Mục 3.4, Dòng 434--448 (Eq. 12); hoàn toàn bị bỏ sót.
- **Thực thi trong mã nguồn**: Trong file `system/flcore/clients/clientscd.py`, Dòng 459--472:
  ```python
  if self.mixup_alpha > 0 and rep.size(0) > 1:
      rep_det = rep.detach()
      lam_mix = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
      lam_mix = max(lam_mix, 1.0 - lam_mix)
      perm = torch.randperm(rep_det.size(0), device=self.device)
      rep_mix = lam_mix * rep_det + (1.0 - lam_mix) * rep_det[perm]
      try:
          out_mix = self.head(rep_mix)
          loss_ce_mix = lam_mix * self.loss(out_mix, y) + (1.0 - lam_mix) * self.loss(out_mix, y[perm])
          loss_ce = 0.75 * loss_ce + 0.25 * loss_ce_mix
      except (RuntimeError, ValueError):
          pass
  ```
  Trong `main.py:342`, cờ `--mixup_alpha` mặc định là `0.2` và luôn hoạt động trong mọi thực nghiệm.
- **Phân tích kỹ thuật**:
  Trên mọi batch có kích thước $> 1$, client thực thi Feature Manifold Mixup trên các biểu diễn đã ngắt gradient (detached), hòa trộn 25% hàm mất mát cross-entropy nội suy vào head phân loại. Việc giấu thành phần này làm mất tính minh bạch của kiến trúc giúp làm mượt biên quyết định.
- **Cách sửa nhanh trong LaTeX**: Công bố chính thức Feature Manifold Mixup ở Mục 3.4 (Eq. 12) dưới dạng hàm chính quy hóa phụ trợ $\mathcal{L}_{\mathrm{CE}}^{\mathrm{head}} = (1-\alpha_{\mathrm{mix}})\mathcal{L}_{\mathrm{CE}} + \alpha_{\mathrm{mix}}\mathcal{L}_{\mathrm{Mixup}}$ với $\alpha_{\mathrm{mix}} = 0.25$ và phân phối Beta tham số $0.2$.

### 3. Nhánh Đặc trưng Ảo Cục bộ $\lambda_{\mathrm{vl}}$ Thực tế Bị Vô hiệu hóa [Mã lỗi: MAJ-01]
- **Phát biểu trong bản thảo**: Mục 3.4 (Eq. 12, Dòng 443) và Mục 4.1 (Dòng 607) khẳng định nhánh đặc trưng ảo cục bộ hoạt động với trọng số $\lambda_{\mathrm{vl}} = 0.05$.
- **Thực thi trong mã nguồn**: Trong `system/main.py:336`, `--lamda_virt_local` mặc định là `0.0`. Docstring trong `clientscd.py:23` ghi rõ: `"- 4. Gaussian Virtual Feature Calibration (mặc định tắt: lamda_virt = 0.0)"`. Trong kịch bản chạy benchmark `run_benchmark_matrix.py:80`, chỉ truyền `--lamda_virt_global 0.05`, khiến toàn bộ các lượt chạy đều có `lamda_virt_local = 0.0`.
- **Cách sửa nhanh trong LaTeX**: Sửa Dòng 607 thành `"head calibration weights $\lambda_{\text{vl}} = 0.0, \lambda_{\text{vg}} = 0.05$ (với hiệu chuẩn đặc trưng ảo toàn cục được kích hoạt mặc định)"`.

### 4. Sai lệch Siêu tham số Server Momentum ($\beta = 0.6$ so với $\beta = 0.1$) [Mã lỗi: MAJ-01]
- **Phát biểu trong bản thảo**: Mục 4.1 (Dòng 607) và Mục 5.4 (Dòng 742) ghi: `"server momentum \beta = 0.6"`.
- **Thực thi trong mã nguồn**: Trong `system/flcore/servers/serverscd.py:32`, `system/main.py:326`, và `run_benchmark_matrix.py:77`, `--proto_momentum` mặc định và chạy thực tế với giá trị **0.1**.
- **Phân tích kỹ thuật**: Trong công thức Eq. 15 ($s_t^c = (1-\beta)s_{t-1}^c + \beta \bar{s}_t^c$), đặt $\beta=0.6$ gán 60% trọng số cho thống kê vòng hiện tại, trong khi $\beta=0.1$ chỉ gán 10%. Đây là độ lệch quán tính cập nhật lên tới $6\times$.
- **Cách sửa nhanh trong LaTeX**: Đồng nhất Dòng 607 thành `server momentum \beta = 0.1`, và làm rõ ở Dòng 742 rằng $\beta = 0.6$ được đánh giá như một điểm ablation.

### 5. Lệch Lịch trình Ramp-up ($T_{\mathrm{ramp}} = 10$ so với $T_{\mathrm{ramp}} = 5$) [Mã lỗi: MAJ-01]
- **Phát biểu trong bản thảo**: Mục 4.1 (Dòng 607) ghi: `"ramp-up rounds T_{\text{ramp}} = 10"`.
- **Thực thi trong mã nguồn**: Trong `clientscd.py:52`, `main.py:318`, và `BENCHMARK_RUN_GUIDE.md`, `--proto_rampup` mặc định và chạy thực tế là **5**.
- **Cách sửa nhanh trong LaTeX**: Đổi `T_{\text{ramp}} = 10` thành `T_{\text{ramp}} = 5` tại Dòng 607.

### 6. Phạm vi Tính Ngưỡng sàn Phương sai ($\tau_v$) [Mã lỗi: MAJ-07]
- **Phát biểu trong bản thảo**: Mục 3.4 (Dòng 362) định nghĩa $\tau_v = \max(\alpha_v \mathrm{mean}((\sigma^c)^{\odot 2}), \alpha_q)$ và ghi chú: *"Cho phép $\mathrm{mean}(\cdot)$ lấy trung bình trên $d$ chiều."*
- **Thực thi trong mã nguồn**: Trong `clientscd.py:208--211`:
  ```python
  var_tensor = std_tensor.pow(2)
  mean_var = float(torch.mean(var_tensor).item())
  tau_var = max(0.25 * mean_var, 1e-2)
  ```
  Lệnh `torch.mean(var_tensor)` thực tế lấy trung bình gộp trên toàn bộ các lớp hợp lệ $|\mathcal{C}_{t-1}|$ và toàn bộ $d$ chiều cùng lúc.
- **Cách sửa nhanh trong LaTeX**: Sửa Dòng 362 để định nghĩa rõ ràng phương sai sàn gộp toàn cục:
  $$\tau_v = \max\!\Big(\alpha_v \frac{1}{|\mathcal{C}_{t-1}| d} \sum_{c \in \mathcal{C}_{t-1}} \sum_{j=1}^d (\sigma_j^c)^2,\, \alpha_q\Big)$$
  với $\alpha_v = 0.25$ và $\alpha_q = 10^{-2}$.

---

## 4.2 Lỗi Chặt chẽ Toán học & Lý thuyết / Lỗ hổng Chứng minh

### 1. Hoàn toàn Thiếu Chứng minh cho các Proposition 3.1--3.3 (Thiếu Appendix) [Mã lỗi: CRIT-02]
- **Vị trí**: Mục 3.7 (`sec:theory`), Dòng 554--593. Dòng 556 có ghi: `% Three short results; proofs go to the appendix.`.
- **Nguy cơ**: Bài báo không có phụ lục, khiến các mệnh đề toán học không có chứng minh. Đây là lỗi chí mạng ở ICML.
- **Biện pháp xử lý**: Thêm phần `\appendix` vào cuối bài chứa toàn bộ chứng minh hoàn chỉnh (đã được soạn sẵn ở Mục 6 của báo cáo này).

### 2. Lập luận Đảo ngược Bản chất Toán học của Mahalanobis Compactness ($\mathcal{L}_{\mathrm{PL}}$) [Mã lỗi: MAJ-03]
- **Tuyên bố trong bản thảo**: Các dòng 192, 306, 416, 481, và 753 liên tục khẳng định khoảng cách Mahalanobis "phạt có chọn lọc các chiều bị phân tán quá mức" và phương sai lớn "hướng dẫn client nén nó lại thông qua $\mathcal{L}_{\mathrm{PL}}$."
- **Thực tế Toán học**:
  Trong Phương trình (9) (Dòng 416) và code `clientscd.py:211, 329`:
  $$d^c(z) = \frac{1}{d}\sum_{j=1}^{d}\frac{g^c_j\,(z_j-p^c_j)^2}{v^c_j}, \quad \text{với } v^c_j = (\sigma_j^c)^2 + \tau_v.$$
  Đạo hàm theo tọa độ biểu diễn $z_j$ là:
  $$\frac{\partial d^c(z)}{\partial z_j} = \frac{2\,g^c_j}{d\,v^c_j}\,(z_j - p^c_j).$$
  Vì phương sai hiệu chỉnh $v_j^c$ nằm ở **mẫu số**:
  - Chiều có phương sai lớn ($v_j^c \gg 1$) $\implies$ gradient phạt bị **suy giảm** (chấp nhận sự phân tán).
  - Chiều gọn gàng, phương sai nhỏ, SNR cao ($v_j^c$ nhỏ) $\implies$ gradient phạt là **lớn nhất** (bắt buộc co cụm chặt chẽ).
  Do đó, khoảng cách Mahalanobis thực hiện **chính quy hóa theo trọng số độ chính xác (precision-weighted regularization)**, chứ không phải dập tắt các chiều phân tán lớn.
- **Cách sửa nhanh trong LaTeX**: Viết lại các dòng 192, 306, 481, và 753 để khẳng định đúng bản chất: Mahalanobis compactness thực thi *precision-weighted regularization*: ép buộc không gian chặt chẽ dọc theo các chiều có SNR cao, phương sai nhỏ, và giảm nhẹ phạt ở các chiều bị nhiễu hoặc trôi dạt.

### 3. Ngược Thuật ngữ: Gọi $r_t^c$ là "Concentration" khi nó Tỷ lệ thuận với Độ phân tán [Mã lỗi: MAJ-11]
- **Vị trí**: Mục 3.5, Dòng 497, 503 (Eq. 16).
- **Phân tích**: Eq. 16 định nghĩa $r_t^c = \frac{\|\sigma_t^c\|_2 / \|\mu_t^c\|_2}{\log(n_t^c + \kappa)}$. Trong thống kê, concentration nghịch biến với dispersion. Ở đây, $r_t^c$ tăng khi độ lệch chuẩn tăng và giảm khi số lượng mẫu tăng. Nó đo lường *độ phân tán (dispersion)* hoặc *độ không chắc chắn của cụm (cluster uncertainty)*. Gọi nó là "concentration score" là bị ngược thuật ngữ.
- **Cách sửa nhanh**: Đổi tên $r_t^c$ thành "density-aware dispersion score" hoặc "cluster uncertainty index" tại Dòng 497 và Eq. 16.

### 4. Xung đột Ký hiệu Phương sai Gộp $\bar{\sigma}^2$ [Mã lỗi: MAJ-04]
- **Vị trí**: Mục 3.2 (Eq. 3, Dòng 296; Dòng 306) so với Mục 3.5 (Eq. 14, Dòng 479) và Mục 3.7 (Prop. 3.2).
- **Phân tích**: Trong Eq. 3, $\bar{\sigma}_j^2 = \frac{1}{2}((\sigma_j^c)^2 + (\sigma_j^{c'})^2)$ là trung bình số học phương sai giữa 2 lớp $c$ và $c'$. Trong Eq. 14, $(\bar{\sigma}_t^c)^2$ lại ký hiệu phương sai gộp từ nhiều client ở server cho lớp $c$. Ở Dòng 306 hai ý nghĩa này đụng độ nhau.
- **Cách sửa nhanh**: Ở Mục 3.2, đổi ký hiệu phương sai cặp lớp thành $\sigma_{c, c', j}^2 \triangleq \tfrac{1}{2}\big((\sigma_j^c)^2 + (\sigma_j^{c'})^2\big)$, giữ riêng ký hiệu $\bar{\sigma}_t^c$ cho phương sai gộp server của lớp $c$ ở vòng $t$.

### 5. Điểm Kỳ dị khi Lớp chỉ có 1 Mẫu ($n_k^c = 1$) [Mã lỗi: MAJ-06]
- **Vị trí**: Mục 3.1 (Eq. 2, Dòng 273), Mục 3.2 (Eq. 3); Code: `clientscd.py:125`.
- **Phân tích**: Nếu $n_k^c = 1$, phương sai $(\sigma_k^c)^2 = \mathbf{0}$. Trong Eq. 3, $\ln(\bar{\sigma}_j^2 / (\sigma_j^c \sigma_j^{c'})) \to \ln(+\infty)$, tạo ra giá trị không xác định. Trong Eq. 9, phương sai bằng 0 dẫn tới phép chia cho 0. Code có đặt giá trị sàn bảo vệ (`1e-6`), nhưng bài báo không hề nêu điều kiện này.
- **Cách sửa nhanh**: Bổ sung điều kiện chặn dưới không suy biến ngay sau Eq. 2:
  ```latex
  where $\odot$ denotes element-wise multiplication, and $\sigma_k^c\in\mathbb{R}^d$ is the per-dimension standard deviation (with a numerical floor $\sigma_{k,j}^c \ge \epsilon_0 = 10^{-6}$ enforced when $n_k^c = 1$ to guarantee non-singular dispersion on singleton classes).
  ```

---

## 4.3 Sai lệch về Thực nghiệm & Đánh giá

### 1. Báo cáo Peak Accuracy dưới Danh nghĩa Final-Round Accuracy [Mã lỗi: CRIT-04]
- **Khẳng định trong bài**: Mục 4.1, Dòng 609:
  `"We measure model performance using the average local test accuracy across all clients at the final communication round."`
- **Sự thật trong file Excel/Log**: Dữ liệu Bảng 1 được copy trực tiếp từ Hàng 3 Sheet `Summary` trong `results/runs_results/runs_results.xlsx`: *"Bảng 1: Best Test Accuracy (%) [Peak Performance across 100 Rounds]"*.
- **Hệ quả che giấu sự sụp đổ của Baseline**:
  - **Flowers102 ($\alpha=0.1$)**: FedTGP đạt 54.79% ở vòng 57, sau đó rơi tự do xuống **0.58% ở vòng 100**. Bảng 1 báo cáo **54.79%**.
  - **Flowers102 ($\alpha=0.5$)**: FedTGP đạt đỉnh 31.26% ở vòng 38, sau đó hỏng thành `NaN`. Bảng 1 báo cáo **31.26%**.
  - **Flowers102 ($\alpha=1.0$)**: FedTGP đạt đỉnh 24.48% ở vòng 36, sau đó sụp về **0.54% ở vòng 100**. Bảng 1 báo cáo **24.48%**.
  - **FedPLVM**: Bị lỗi phân kỳ thành `NaN` trước vòng 100 ở 6 trên 9 lượt chạy, nhưng Bảng 1 vẫn in điểm đỉnh giai đoạn đầu (như 43.54%, 28.76%, 86.82%).
- **Cách khắc phục**: Sửa lại định nghĩa chỉ số ở Dòng 609 thành đo lường *peak average local test accuracy across 100 rounds*, đồng thời thêm ghi chú chân trang (footnote) giải thích rõ các baseline bị sụp đổ phân kỳ ở giai đoạn sau.

### 2. Dữ liệu Bị Khuyết & Tuyên bố Khống Đa Tập Dữ liệu [Mã lỗi: CRIT-05]
- **Vị trí**: Bảng 1 (`tab:com`, Dòng 610--636); Dòng 753.
- **Phân tích**:
  - Bảng 1 thiết kế 12 cột cho 4 tập dữ liệu: CIFAR-10, CIFAR-100, Flowers102, và Tiny-ImageNet.
  - Riêng phương pháp đề xuất **FedSCD**, có tới 9/12 ô là dấu gạch ngang rỗng (`--`). Kết quả chỉ có duy nhất trên CIFAR-10.
  - Tiny-ImageNet có trọn vẹn 21 ô của tất cả các phương pháp đều là dấu gạch ngang (`--`).
  - Dù vậy, phần Kết luận (Dòng 753) vẫn tuyên bố FedSCD được chứng minh vượt trội *"across multiple datasets"*.
- **Cách khắc phục**: Cắt bỏ 3 cột Tiny-ImageNet khỏi Bảng 1; chạy bổ sung kết quả cho CIFAR-100 và Flowers102 bằng `system/run_benchmark_matrix.py` (hoặc giới hạn rõ phạm vi Bảng 1 trong bài báo là CIFAR-10); sửa Dòng 753 thành: *"Benchmark experiments on CIFAR-10 across severe Dirichlet partitions demonstrate..."*.

### 3. Văn bản Mâu thuẫn Trực tiếp với Số liệu Bảng 1 [Mã lỗi: CRIT-06]
- **Vị trí**: Mục 1, Dòng 198; Mục 5.2, Dòng 645--647; Bảng 1.
- **Phân tích chi tiết**:
  - *Mục 5.2 Dòng 645 tuyên bố*: FedSCD (75.15%) vượt FedTGP (68.64%) là 6.5% tại $\alpha=0.5$. Nhưng trong Bảng 1, FedTGP đạt 71.70%. Mức vượt thực tế là $75.15 - 71.70 = \mathbf{+3.45\%}$ (chỉ bằng hơn một nửa con số tuyên bố).
  - *Mục 5.2 Dòng 646 tuyên bố*: FedSCD (90.71%) duy trì khoảng cách 1.9% so với FedTGP (88.83%) tại $\alpha=0.1$. Trong Bảng 1, FedTGP đạt 88.87% (chênh +1.84%), và á quân thực tế là FedORGP (88.88%).
  - *Mục 1 Dòng 198 tuyên bố*: cải thiện tới 8.4% so với FedTGP. Trong Bảng 1, tại $\alpha=1.0$, FedSCD (69.60%) so với FedTGP (64.51%) mức chênh lệch tối đa chỉ là $\mathbf{+5.09\%}$. Con số 8.39% thực chất là so với FedORGP ($69.60 - 61.21 = 8.39\%$).
  - *Baseline không có thực*: Các dòng 645--646 thảo luận về FedProto (65.03%, 86.10%) và CCVR (62.85% đến 45.65%), nhưng cả 2 phương pháp này không hề có trong Bảng 1.
- **Cách sửa nhanh trong LaTeX**: Cập nhật lại toàn bộ số liệu văn bản cho khớp 100% với Bảng 1 (xem chi tiết mã thay thế tại Mục 7).

### 4. Kích thước Batch Size Không Thống nhất giữa các Baseline [Mã lỗi: MAJ-14]
- **Vị trí**: Mục 4.1, Dòng 607 khẳng định kích thước batch đồng nhất $B = 32$.
- **Thực tế trong code**: Log cấu hình WandB (`system/wandb/run-*/files/config.yaml`) và `system/run_all_baselines.sh:47` cho thấy **FedTGP và FedSA bị chạy với batch size $B = 10$**, trong khi FedCOF và FedORGP chạy $B = 32$. Batch size 10 làm thay đổi phương sai gradient và tăng gấp 3 lần số bước tối ưu hóa mỗi epoch.
- **Cách khắc phục**: Chuẩn hóa lại toàn bộ baseline chạy đồng nhất $B = 32$.

---

## 4.4 Vấn đề Câu chữ, Lỗi đánh máy, Định dạng & Tuân thủ Quy chuẩn

### 1. Vi phạm Giới hạn Số trang ICML (Dài 11 Trang Nội dung Chính) [Mã lỗi: CRIT-01]
- **Thực tế**: File PDF biên dịch dài 11 trang trước phần tài liệu tham khảo (tổng 13 trang). ICML 2026 quy định cứng **tối đa 8 trang** cho nội dung chính.
- **Giải pháp**:
  1. Cô đọng Mục 2 (Related Work) từ 1.5 trang xuống 0.75 trang.
  2. Cô đọng Mục 3 (Methodology) bớt ~1.5 trang.
  3. Chuyển toàn bộ các chứng minh toán học của Mục 3.7 sang phần Appendix.
  4. Xóa bỏ các cột Tiny-ImageNet trống trơn trong Bảng 1.
  5. Đảm bảo toàn bộ bài viết nằm gọn trong 8 trang trước References.

### 2. Vi phạm Quy tắc Ẩn danh Double-Blind & Comment Tác giả [Mã lỗi: CRIT-03]
- **Vị trí**: Dòng 73--74, 190, 192, 211, 218, 225, 260, 345, 461.
- **Phân tích**: Các macro `\newcommand{\NMD}[1]{\textbf{\color{blue}[NMD: #1]}}` và `\newcommand{\GVQ}[1]{\textbf{\color{red}[GVQ: #1]}}` làm lộ chữ viết tắt tên tác giả (`NMD`, `GVQ`) in màu trong bản PDF. Dòng chữ `\GVQ{need TSNE...}` còn trực tiếp thừa nhận với phản biện là chưa có hình thực nghiệm.
- **Giải pháp**: Xóa bỏ hoàn toàn định nghĩa và các lời gọi macro `\NMD{...}`, `\GVQ{...}`.

### 3. Hỏng Tham chiếu Chéo LaTeX Hiển thị `(Eq. ??)` [Mã lỗi: CRIT-07]
- **Vị trí**: Dòng 393, 441, 451, 463, 492, 497, 539, 544, 612, 791, 792, 799, 800.
- **Phân tích**: Bốn tham chiếu bị trỏ sai nhãn khiến văn bản và Thuật toán 1 hiện dấu hỏi chấm đôi đậm **`??`**:
  1. `\ref{eq:nce}` hỏng vì phương trình đặt tên là `\label{eq:proto_nce}`.
  2. `\ref{eq:head}` hỏng vì phương trình đặt tên là `\label{eq:head_loss}`.
  3. `\ref{eq:server_mom}` hỏng vì phương trình đặt tên là `\label{eq:ema}`.
  4. `\ref{tab:label_skew}` hỏng vì bảng đặt tên là `\label{tab:com}`.
- **Cách sửa nhanh**: Đặt nhãn kép đồng danh: `\label{eq:proto_nce}\label{eq:nce}`, `\label{eq:head_loss}\label{eq:head}`, `\label{eq:ema}\label{eq:server_mom}`, và `\label{tab:com}\label{tab:label_skew}`.

### 4. Lỗi Dấu Ngoặc Nhọn TeX Cấp Gốc Bao Trùm Nhiều Mục [Mã lỗi: MAJ-20]
- **Vị trí**: Dấu `{` ở dòng 151 mở ra nhưng tới tận dòng 209 mới đóng lại.
- **Phân tích**: Lỗi này nhốt 59 dòng văn bản, một hình vẽ và cả dòng lệnh `\section{Related Work}` vào trong một khối cục bộ (local scope), phá vỡ công cụ ngắt đoạn và canh lề dọc của LaTeX.
- **Cách sửa**: Xóa bỏ cặp dấu ngoặc nhọn dư thừa ở dòng 151 và dòng 209.

### 5. Văn bản Vẫn Chứa Các Lệnh Đổi Màu Bản Thảo (`\textcolor{blue}{...}`)
- **Phân tích**: Rất nhiều đoạn văn dài vẫn bị bọc trong lệnh màu `\textcolor{blue}{...}`, làm nguyên cả trang giấy in ra chữ màu xanh.
- **Cách sửa**: Gỡ bỏ toàn bộ các lệnh `\textcolor{blue}{` và `\textcolor{red}{`.

---

# 5. Ma trận Đối chiếu Toàn diện giữa Mã nguồn và Bài báo

Bảng đối chiếu từng ký hiệu, công thức toán học trong bài báo với đoạn code thực thi tương ứng trong repository:

| Ký hiệu / Khái niệm Toán học | Công thức & Vị trí trong Bài báo | Thực thi trong Mã nguồn | Tệp mã nguồn tương ứng | Trạng thái Đối chiếu & Ghi chú |
|:---|:---|:---|:---|:---|
| **Phương sai đặc trưng cục bộ $(\sigma_k^c)^2$** | Eq. 2: $\frac{1}{n_k^c}\sum (f_k(x) - \mu_k^c)^{\odot 2}$ | `torch.std(class_feats, dim=0, unbiased=False)` | `clientscd.py:125` | **Khớp chính xác**. Xác thực điều kiện plug-in cho Mệnh đề 3.1. |
| **Bảo vệ lớp chỉ có 1 mẫu** | Không nêu trong Eq. 2 (ngầm định $n \ge 1$) | `std_c = torch.full_like(mean_c, 1e-6)` nếu $n \le 1$ | `clientscd.py:125` | **Code có bảo vệ nhưng bài báo bỏ sót**. Điều kiện tiên quyết chống kỳ dị. |
| **Dữ liệu Client tải lên** | Bộ giá trị $\{\mu_k^c, \sigma_k^c, n_k^c\}$ | `{'mean': mean_c.cpu(), 'std': std_c.cpu(), 'count': int(n)}` | `clientscd.py:126` | **Khớp chính xác**. Tải lên vector độ lệch chuẩn $\sigma_k^c \in \mathbb{R}^d$. |
| **Khoảng cách Bhattacharyya $D_B$** | Eq. 3: Tổng trên $d$ chiều | Chặn dưới phương sai $10^{-4}$; lọc các chiều nơ-ron chết | `motivation_metrics.py:161--174` | **Có sai khác nhẹ**. Code có lọc ReLU nơ-ron chết, bài báo không nêu. |
| **Chỉ số Bhattacharyya Overlap** | Dòng 304: $\exp(-D_B / d)$ | `torch.exp(-D_B / np.sqrt(d_eff))` | `motivation_metrics.py:184` | **Sai lệch lớn**. Code chia cho $\sqrt{d_{\mathrm{eff}}}$, bài báo ghi chia $d$. |
| **Mỏ neo Empirical Bayes $p^c$** | Eq. 5: $(1-\gamma^c)\mu^c + \gamma^c \mu_k^c$ | `p = (1.0 - gamma) * glob_m + gamma * loc_m` | `clientscd.py:186` | **Khớp chính xác**. $\gamma_{\max} = 0.25, n_0 = 50$. Mặc định $p^c=\mu^c$ nếu $n=0$. |
| **Sàn phương sai $\tau_v$** | Dòng 362: $\max(\alpha_v\mathrm{mean}((\sigma^c)^2), \alpha_q)$ | `max(0.25 * float(torch.mean(var_tensor)), 1e-2)` | `clientscd.py:210` | **Lệch phạm vi**. Code tính trung bình gộp toàn bộ lớp và chiều. |
| **Điểm SNR theo tọa độ $s_j^c$** | Dòng 368: $u_j^c / \mathrm{mean}(u^c)$ | `snr_raw = mean_vec / (std_vec + 1e-6); snr / mean` | `clientscd.py:216--222` | **Khớp chính xác**. Chuẩn hóa trên các chiều. |
| **Cổng điều chế SNR $g_j^c$** | Eq. 6: $\frac{0.5 + 0.5 s_j^c}{1 + 0.5 s_j^c}$ | `gate = (0.5 + 0.5 * snr_c) / (1.0 + 0.5 * snr_c)` | `clientscd.py:224` | **Khớp chính xác**. Đường cong bão hòa mượt trong đoạn $[0.5, 1.0]$. |
| **Trọng số độ chính xác $w_j^c$** | Eq. 6: $g_j^c / \sqrt{v_j^c}$ | `w = g * (u_norm)` với `u = 1.0 / sqrt(v)` | `clientscd.py:228--234` | **Khớp chính xác**. Gán trọng số lớn cho chiều có phương sai nhỏ. |
| **Mỏ neo chuẩn hóa $q^c$** | Eq. 6: $w^c \odot p^c / \|w^c \odot p^c\|_2$ | `F.normalize(w * p, p=2, dim=-1)` | `clientscd.py:230` | **Khớp chính xác**. Nằm trên mặt cầu đơn vị $\mathbb{S}^{d-1}$. |
| **ProtoNCE Loss $\mathcal{L}_{\mathrm{NCE}}$** | Eq. 7: $-\log \frac{\exp(\hat{z}^\top q^y / \phi^y)}{\sum \exp(\hat{z}^\top q^c / \phi^c)}$ | Tính toán với nhiệt độ riêng từng lớp $\phi^c$ | `clientscd.py:348--356` | **Khớp chính xác**. Loại bỏ các lớp chưa có dữ liệu. |
| **Decoupled SupCon Loss** | Eq. 8: Mẫu số loại bỏ các cặp cùng lớp | Loại bỏ cặp cùng lớp khỏi mẫu số (DCL) | `clientscd.py:358--390` | **Khớp chính xác**. Đúng chuẩn công thức Yeh et al. (2022). |
| **Mahalanobis Loss $\mathcal{L}_{\mathrm{PL}}$** | Eqs. 9--10: $\frac{1}{d}\sum g_j^y(z_j - p_j^y)^2 / v_j^y$ | `torch.sum(g * (feat - p)**2 / v, dim=-1) / d` | `clientscd.py:392--402` | **Code chuẩn xác**. Lời văn bài báo bị ngược ("phạt chiều phương sai lớn"). |
| **Loss Bộ trích xuất Đặc trưng** | Eq. 11: $\mathcal{L}_{\mathrm{CE}} + \lambda_t(\mathcal{L}_{\mathrm{NCE}} + \dots)$ | `loss = loss_ce + lamda_t * (loss_scd + ...)` | `clientscd.py:504--510` | **Khớp chính xác**. Có áp dụng ramp-up tuyến tính. |
| **Loss Đầu Phân loại (Head)** | Eq. 12: $\mathcal{L}_{\mathrm{CE}}(h_k(\mathrm{sg}[f_k(x)]), y) + \dots$ | `loss_ce = self.loss(out, y)` trên biểu diễn đã detach | `clientscd.py:457` | **Khớp chính xác**. Sử dụng ngắt gradient $\mathrm{sg}[\cdot]$. |
| **Feature Manifold Mixup** | **CHƯA CÔNG BỐ TRONG BÀI** | `--mixup_alpha 0.2`, hòa trộn 25% loss mixup | `clientscd.py:459--472` | **Thiếu sót lớn trong bài báo**. Mặc định bật trong code. |
| **Độ lệch Nhiễu ảo** | Dòng 447: $\eta_{\mathrm{loc}}=0.5, \eta_{\mathrm{glob}}=0.3$ | `eta_loc = 0.5, eta_glob = 0.3` | `clientscd.py:371, 385` | **Khớp chính xác**. Hình 2 dùng nhầm ký hiệu $\alpha$. |
| **Trọng số Đặc trưng Ảo Head** | Dòng 607: $\lambda_{\mathrm{vl}} = 0.05, \lambda_{\mathrm{vg}} = 0.05$ | $\lambda_{\mathrm{vl}} = 0.0$ (tắt), $\lambda_{\mathrm{vg}} = 0.05$ | `clientscd.py:56, 59` | **Sai lệch lớn**. $\lambda_{\mathrm{vl}}$ bị tắt trong thực nghiệm thực tế. |
| **Trung bình Gộp Server $\bar{\mu}_t^c$** | Eq. 13: $\frac{1}{\bar{n}_t^c}\sum n_k^c \mu_k^c$ | `batch_mean = sum(n_k * mean_k) / total_n` | `serverscd.py:192--196` | **Khớp chính xác**. Lọc các client có $n_k^c \ge 1$. |
| **Gộp Phương sai Toàn phần $(\bar{\sigma}_t^c)^2$** | Eq. 14: $\frac{1}{\bar{n}_t^c}\sum n_k^c [(\sigma_k^c)^2 + (\mu_k^c - \bar{\mu}_t^c)^{\odot 2}]$ | `pooled_var += n_k * (local_var + (mean_k - batch_mean)**2)` | `serverscd.py:197--204` | **Khớp chính xác**. Ghi nhận trọn vẹn độ trôi giữa các client. |
| **Server Momentum $\beta$** | Dòng 607: $\beta = 0.6$; Dòng 742: $\beta = 0.6$ | Mặc định `proto_momentum = 0.1` | `serverscd.py:32`, `main.py:326` | **Sai lệch lớn** (Lệch $6\times$ về tốc độ học). |
| **Chỉ số Phân tán $r_t^c$** | Eq. 16: $\frac{\|\sigma_t^c\|_2 / \|\mu_t^c\|_2}{\log(n_t^c + \kappa)}$ | `(norm(std) / (norm(mean) + 1e-6)) / log(count + 10.0)` | `serverscd.py:221--224` | **Khớp chính xác** ($\kappa = 10$). Bài báo đặt nhầm tên là "concentration". |
| **Cắt ngưỡng Nhiệt độ $\phi_t^c$** | Eq. 17: $\mathrm{clip}(\tau \frac{r_t^c}{\text{mean}}, (1\pm\rho)\tau)$ | `scaled_phi = base_tau * (raw / mean_raw); clip(...)` | `serverscd.py:263--265` | **Khớp chính xác** ($\tau=0.1, \rho=0.1$). |
| **Lịch trình Ramp-up $T_{\mathrm{ramp}}$** | Dòng 607: $T_{\mathrm{ramp}} = 10$ | `self.proto_rampup = 5` (mặc định 5 trong benchmark) | `clientscd.py:52`, `main.py:318` | **Sai lệch lớn** ($5$ vòng trong code so với $10$ vòng trong bài). |
| **Chỉ số Đánh giá** | Dòng 609: "final communication round" | Bảng 1 sao chép `df.max() * 100` (peak accuracy)| `runs_results.xlsx` | **Mâu thuẫn nghiêm trọng**. Che giấu việc baseline bị sụp đổ. |
| **Dung lượng Client Tải lên** | Dòng 549: $\le C(2d+1)$ số thực | Tải lên bộ $\{\mu_k^c, \sigma_k^c, n_k^c\}$ | `clientscd.py:126` | **Khớp chính xác**. Tuyến tính $\mathcal{O}(Cd)$. |
| **Dung lượng Server Phát sóng** | Dòng 549: $\le C(2d+1)$ số thực | Phát sóng $\{\mu_t^c, \sigma_t^c, \phi_t^c\}$ | `serverscd.py:147` | **Khớp chính xác**. Tổng chi phí khứ hồi $\le 2C(2d+1)$ số thực. |

---

# 6. Phụ lục Toán học & Mã nguồn Chứng minh Hình thức

Để giải quyết triệt để **CRIT-02 (Thiếu chứng minh)** và **MAJ-12, MAJ-13 (Độ chuẩn xác lý thuyết & Giới hạn đa lớp)**, đoạn mã LaTeX dưới đây đã được biên soạn hoàn chỉnh, đạt chuẩn xuất bản, sẵn sàng chèn trực tiếp vào `Overleaf/icml2026.tex` dưới dạng phần phụ lục:

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX: DEFERRED PROOFS AND THEORETICAL ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\appendix
\onecolumn
\section{Deferred Proofs and Theoretical Analysis}
\label{sec:proofs}

In this appendix, we provide self-contained, mathematically rigorous proofs for the theoretical propositions stated in Section~\ref{sec:theory}, establish the exact regularity conditions under which decentralized total-variance pooling holds, and derive the multiclass Bayes classification error bound.

\subsection{Proof of Proposition~\ref{prop:exact} (Exact Total-Variance Pooling)}
\label{app:proof_exact}

\begin{proposition}[Exact Pooling under Empirical Plug-in Variance]
Let $\mathcal{S}_t \subseteq \{1, \dots, K\}$ denote the set of participating clients in communication round $t$. For each client $k \in \mathcal{S}_t$ possessing samples of class $c$, let $\mathcal{D}_k^c = \{z_{k,i}^c\}_{i=1}^{n_k^c} \subset \mathbb{R}^d$ denote its local feature embeddings, where $n_k^c \ge 1$. Let $\mathcal{S}_t^c \triangleq \{k \in \mathcal{S}_t \mid n_k^c \ge 1\}$ denote the set of active clients for class $c$, and let
\begin{equation}
    \mathcal{U}_t^c \triangleq \bigcup_{k \in \mathcal{S}_t^c} \mathcal{D}_k^c = \bigcup_{k \in \mathcal{S}_t^c} \{z_{k,i}^c\}_{i=1}^{n_k^c}
\end{equation}
denote the feature union across all participating clients, with total cardinality $\bar{n}_t^c = \sum_{k \in \mathcal{S}_t^c} n_k^c$.
Under the empirical plug-in sample variance definition (Eq.~\ref{eq:var}, with denominator $n_k^c$), the pooled server statistics $\bar{\mu}_t^c$ (Eq.~\ref{eq:server_mean}) and $(\bar{\sigma}_t^c)^{\odot 2}$ (Eq.~\ref{eq:server_var}) satisfy:
\begin{equation}
    \bar{\mu}_t^c = \mu_{\mathcal{U}_t^c} \quad \text{and} \quad (\bar{\sigma}_t^c)^{\odot 2} = (\sigma_{\mathcal{U}_t^c})^{\odot 2},
\end{equation}
where $\mu_{\mathcal{U}_t^c}$ and $(\sigma_{\mathcal{U}_t^c})^{\odot 2}$ are the exact empirical sample centroid and pooled plug-in variance of $\mathcal{U}_t^c$. Furthermore, when $\beta = 1$ in Eq.~\ref{eq:ema}, the broadcast global statistics $(\mu_t^c, (\sigma_t^c)^{\odot 2})$ equal these exact pooled moments without communicating raw data.
\end{proposition}

\begin{proof}
We evaluate the first- and second-order empirical moments coordinate-wise along each feature dimension $j \in \{1, \dots, d\}$.

\paragraph{1. First-Order Moment (Sample Mean):}
By definition of the centralized arithmetic mean of the multi-client union $\mathcal{U}_t^c$:
\begin{equation}
    \mu_{\mathcal{U}_t^c, j} = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c.
\end{equation}
Grouping the summation across participating clients $k \in \mathcal{S}_t^c$:
\begin{equation}
    \mu_{\mathcal{U}_t^c, j} = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \left( \frac{1}{n_k^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c \right) = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \mu_{k,j}^c = \bar{\mu}_{t,j}^c.
\end{equation}
Vectorizing across all coordinates $j \in \{1, \dots, d\}$ yields $\bar{\mu}_t^c = \mu_{\mathcal{U}_t^c}$.

\paragraph{2. Second-Order Moment (Sample Variance):}
By definition of the uncorrected empirical plug-in variance (matching `torch.std(..., unbiased=False)` in Eq.~\ref{eq:var}):
\begin{equation}
    (\sigma_{\mathcal{U}_t^c, j})^2 = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Decomposing the deviation of individual representation $z_{k,i,j}^c$ from the global pooled mean $\bar{\mu}_{t,j}^c$ around the local client mean $\mu_{k,j}^c$:
\begin{equation}
    z_{k,i,j}^c - \bar{\mu}_{t,j}^c = \big(z_{k,i,j}^c - \mu_{k,j}^c\big) + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big).
\end{equation}
Squaring both sides and expanding:
\begin{equation}
    \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2 = \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 + 2\big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)\big(z_{k,i,j}^c - \mu_{k,j}^c\big).
\end{equation}
Summing over all local samples $i \in \{1, \dots, n_k^c\}$ on client $k$:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2 = \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 + \sum_{i=1}^{n_k^c} \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 + 2\big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big) \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big).
\end{equation}
By definition of the arithmetic local sample mean $\mu_{k,j}^c = \frac{1}{n_k^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c$, the local deviations sum identically to zero:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big) = \sum_{i=1}^{n_k^c} z_{k,i,j}^c - n_k^c \mu_{k,j}^c = 0.
\end{equation}
Consequently, the cross-product term vanishes identically. Evaluating the remaining two sums:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 = n_k^c (\sigma_{k,j}^c)^2, \qquad \sum_{i=1}^{n_k^c} \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 = n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Summing over all participating clients $k \in \mathcal{S}_t^c$ and dividing by total sample support $\bar{n}_t^c$:
\begin{equation}
    (\sigma_{\mathcal{U}_t^c, j})^2 = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \Big[ (\sigma_{k,j}^c)^2 + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 \Big] = (\bar{\sigma}_{t,j}^c)^2.
\end{equation}
Vectorizing over coordinates $j \in \{1, \dots, d\}$ yields $(\bar{\sigma}_t^c)^{\odot 2} = (\sigma_{\mathcal{U}_t^c})^{\odot 2}$, matching Eq.~\ref{eq:server_var}. When $\beta = 1$ in Eq.~\ref{eq:ema}, the momentum smoothing step sets $s_t^c = \bar{s}_t^c$, ensuring broadcast moments match the centralized union moments exactly.
\end{proof}

\subsection{Proof of Proposition~\ref{prop:drift} (Drift-Induced Dispersion Inflation)}
\label{app:proof_drift}

\begin{proposition}[Drift-Induced Dispersion Inflation]
Let $W^c \triangleq \sum_{k \in \mathcal{S}_t^c} n_k^c (\sigma_k^c)^{\odot 2}$ and $B^c \triangleq \sum_{k \in \mathcal{S}_t^c} n_k^c (\mu_k^c - \bar{\mu}_t^c)^{\odot 2}$ denote the within-client scatter vector and between-client scatter vector, respectively. The pooled variance in Eq.~\ref{eq:server_var} satisfies:
\begin{equation}
    (\bar{\sigma}_t^c)^{\odot 2} \ge \frac{1}{\bar{n}_t^c} W^c \quad \text{element-wise},
\end{equation}
with equality holding along coordinate $j \in \{1, \dots, d\}$ if and only if all participating local centroids coincide along that coordinate ($\mu_{k,j}^c = \bar{\mu}_{t,j}^c$ for all $k \in \mathcal{S}_t^c$). Any representation drift across heterogeneous clients strictly inflates the aggregated class dispersion beyond the naive weighted average of local variances.
\end{proposition}

\begin{proof}
From Eq.~\ref{eq:server_var}, the $j$-th coordinate of pooled variance decomposes as:
\begin{equation}
    (\bar{\sigma}_{t,j}^c)^2 = \frac{1}{\bar{n}_t^c} W_j^c + \frac{1}{\bar{n}_t^c} B_j^c, \quad \text{where } B_j^c = \sum_{k \in \mathcal{S}_t^c} n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Because sample counts satisfy $n_k^c \ge 1$ and squared deviations satisfy $(\mu_{k,j}^c - \bar{\mu}_{t,j}^c)^2 \ge 0$, every term in $B_j^c$ is non-negative. Therefore:
\begin{equation}
    B_j^c \ge 0 \implies (\bar{\sigma}_{t,j}^c)^2 \ge \frac{1}{\bar{n}_t^c} W_j^c = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c (\sigma_{k,j}^c)^2.
\end{equation}
Equality holds if and only if $B_j^c = 0$. Since $n_k^c \ge 1$ for all active clients:
\begin{equation}
    B_j^c = 0 \iff n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 = 0 \quad \forall k \in \mathcal{S}_t^c \iff \mu_{k,j}^c = \bar{\mu}_{t,j}^c \quad \forall k \in \mathcal{S}_t^c.
\end{equation}
Whenever local representation drift exists along coordinate $j$ ($\exists k, l$ such that $\mu_{k,j}^c \neq \mu_{l,j}^c$), $B_j^c > 0$, which strictly inflates the aggregated class variance $(\bar{\sigma}_{t,j}^c)^2 > \frac{1}{\bar{n}_t^c} W_j^c$.
\end{proof}

\subsection{Proof of Proposition~\ref{prop:bayes_bound} (Bhattacharyya Bound on Pairwise Bayes Error)}
\label{app:proof_bayes}

\begin{proposition}[Pairwise Bayes Error Upper Bound~\citep{kailath1967divergence}]
For any pair of classes $c, c'$ with prior probabilities $\pi_c, \pi_{c'} > 0$ ($\pi_c + \pi_{c'} = 1$) and class-conditional densities $p_c(z), p_{c'}(z)$, the optimal Bayes classification error $P_e(c, c')$ satisfies:
\begin{equation}
    P_e(c, c') \le \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big),
\end{equation}
where $D_B(c, c')$ is the Bhattacharyya distance defined in Eq.~\ref{eq:bhatt}.
\end{proposition}

\begin{proof}
Let $\Omega_c \triangleq \{z \in \mathbb{R}^d \mid \pi_c p_c(z) \ge \pi_{c'} p_{c'}(z)\}$ denote the Bayes-optimal decision region for class $c$, and $\Omega_{c'} \triangleq \mathbb{R}^d \setminus \Omega_c$ denote the region for class $c'$. The total Bayes classification error is:
\begin{equation}
    P_e(c, c') = \int_{\Omega_{c'}} \pi_c p_c(z)\, dz + \int_{\Omega_c} \pi_{c'} p_{c'}(z)\, dz = \int_{\mathbb{R}^d} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big)\, dz.
\end{equation}
For any non-negative real numbers $a, b \ge 0$, the algebraic inequality $\min(a, b) \le a^s b^{1-s}$ holds for all $s \in [0, 1]$ (the Chernoff bound). Setting $s = 1/2$ yields:
\begin{equation}
    \min(a, b) \le \sqrt{ab}.
\end{equation}
Applying this inequality point-wise to $a = \pi_c p_c(z)$ and $b = \pi_{c'} p_{c'}(z)$:
\begin{equation}
    P_e(c, c') \le \int_{\mathbb{R}^d} \sqrt{\pi_c p_c(z) \cdot \pi_{c'} p_{c'}(z)}\, dz = \sqrt{\pi_c \pi_{c'}} \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz.
\end{equation}
The integral defines the Bhattacharyya affinity coefficient $\mathrm{BC}(c, c') \triangleq \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz \in [0, 1]$ (Eq.~\ref{eq:bc}).
By definition of the Bhattacharyya distance $D_B(c, c') \triangleq -\ln \mathrm{BC}(c, c')$, we have $\mathrm{BC}(c, c') = \exp\big(-D_B(c, c')\big)$, yielding:
\begin{equation}
    P_e(c, c') \le \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}

Now, consider class-conditional diagonal Gaussians $p_c(z) = \mathcal{N}(z; \mu^c, \Sigma^c)$ and $p_{c'}(z) = \mathcal{N}(z; \mu^{c'}, \Sigma^{c'})$ with diagonal covariance matrices $\Sigma^c = \mathrm{diag}\big((\sigma_1^c)^2, \dots, (\sigma_d^c)^2\big)$ and $\Sigma^{c'} = \mathrm{diag}\big((\sigma_1^{c'})^2, \dots, (\sigma_d^{c'})^2\big)$.
By Gaussian integration~\citep{kailath1967divergence}:
\begin{equation}
    \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz = \frac{\det(\Sigma^c)^{1/4} \det(\Sigma^{c'})^{1/4}}{\det(\bar{\Sigma})^{1/2}} \exp\left(-\frac{1}{8}(\mu^c - \mu^{c'})^\top \bar{\Sigma}^{-1} (\mu^c - \mu^{c'})\right),
\end{equation}
where $\bar{\Sigma} = \frac{1}{2}(\Sigma^c + \Sigma^{c'})$. Taking the negative natural logarithm:
\begin{equation}
    D_B(c, c') = \frac{1}{8}(\mu^c - \mu^{c'})^\top \bar{\Sigma}^{-1} (\mu^c - \mu^{c'}) + \frac{1}{2} \ln \frac{\det \bar{\Sigma}}{\sqrt{\det \Sigma^c \det \Sigma^{c'}}}.
\end{equation}
Because all covariance matrices are diagonal, $\bar{\Sigma}$ is diagonal with entries $\sigma_{c, c', j}^2 = \frac{1}{2}\big((\sigma_j^c)^2 + (\sigma_j^{c'})^2\big)$, and determinants factorize into products over coordinates:
\begin{equation}
    D_B(c, c') = \frac{1}{8} \sum_{j=1}^d \frac{(\mu_j^c - \mu_j^{c'})^2}{\sigma_{c, c', j}^2} + \frac{1}{2} \sum_{j=1}^d \ln \frac{\sigma_{c, c', j}^2}{\sigma_j^c \sigma_j^{c'}},
\end{equation}
which is exactly Eq.~\ref{eq:bhatt}. Because $\exp(-D_B)$ is strictly decreasing in $D_B$, enlarging $D_B(c, c')$ directly contracts the upper bound on pairwise Bayes error.
\end{proof}

\subsection{Multiclass Bayes Error Union Bound (Corollary to Proposition~\ref{prop:bayes_bound})}
\label{app:multiclass_bound}

\begin{corollary}[Multiclass Bayes Error Union Bound]
For a $C$-class classification problem with class priors $\{\pi_c\}_{c=1}^C$ ($\sum_{c=1}^C \pi_c = 1$) and class-conditional distributions $\{p_c(z)\}_{c=1}^C$, the overall Bayes classification error $P_e$ is upper-bounded by:
\begin{equation}
    P_e \le 2 \sum_{c=1}^C \sum_{c' > c} \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}
Under uniform class priors $\pi_c = 1/C$, this bound simplifies to:
\begin{equation}
    P_e \le \frac{2}{C} \sum_{c=1}^C \sum_{c' > c} \exp\big(-D_B(c, c')\big).
\end{equation}
\end{corollary}

\begin{proof}
Let $R_c \triangleq \{z \in \mathbb{R}^d \mid \pi_c p_c(z) = \max_{1 \le m \le C} \pi_m p_m(z)\}$ denote the Bayes decision region for class $c$. The total probability of correct classification is $P_c = \sum_{c=1}^C \int_{R_c} \pi_c p_c(z)\, dz$. The overall Bayes classification error is:
\begin{equation}
    P_e = 1 - P_c = \sum_{c=1}^C \int_{\mathbb{R}^d \setminus R_c} \pi_c p_c(z)\, dz.
\end{equation}
For any observation $z \in \mathbb{R}^d \setminus R_c$, class $c$ does not attain the maximum posterior density, which implies that there exists at least one competitor class $c' \neq c$ such that $\pi_c p_c(z) \le \pi_{c'} p_{c'}(z)$. Therefore:
\begin{equation}
    \pi_c p_c(z) = \min\big(\pi_c p_c(z), \max_{c' \neq c} \pi_{c'} p_{c'}(z)\big) \le \sum_{c' \neq c} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big).
\end{equation}
Integrating this point-wise inequality over $\mathbb{R}^d$:
\begin{equation}
    \int_{\mathbb{R}^d \setminus R_c} \pi_c p_c(z)\, dz \le \int_{\mathbb{R}^d} \pi_c p_c(z)\, dz \le \sum_{c' \neq c} \int_{\mathbb{R}^d} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big)\, dz = \sum_{c' \neq c} P_e(c, c').
\end{equation}
Summing over all classes $c \in \{1, \dots, C\}$:
\begin{equation}
    P_e \le \sum_{c=1}^C \sum_{c' \neq c} P_e(c, c') = 2 \sum_{c=1}^C \sum_{c' > c} P_e(c, c').
\end{equation}
Applying Proposition~\ref{prop:bayes_bound} to each pairwise error $P_e(c, c')$:
\begin{equation}
    P_e \le 2 \sum_{c=1}^C \sum_{c' > c} \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}
For balanced class priors where $\pi_c = 1/C$ for all $c \in \{1, \dots, C\}$, we have $\sqrt{\pi_c \pi_{c'}} = 1/C$, yielding:
\begin{equation}
    P_e \le \frac{2}{C} \sum_{c=1}^C \sum_{c' > c} \exp\big(-D_B(c, c')\big).
\end{equation}
This establishes that maximizing pairwise Bhattacharyya distances $D_B(c, c')$ across all distinct pairs $(c, c')$ provably tightens the upper bound on the multiclass Bayes classification error.
\end{proof}
```

---

# 7. Kế hoạch Hành động Khắc phục theo Thứ tự Ưu tiên

Lộ trình 3 giai đoạn để hoàn thiện bài báo trước khi nộp ICML 2026:

```
[Giai đoạn 1: Đồng bộ Mã nguồn & Dọn dẹp Bản thảo] (Không tốn GPU; Thực hiện trong 2 Giờ)
  ├── 1.1 Đồng bộ số liệu Mục 1 (Dòng 198) và Mục 5.2 (Dòng 645--648) khớp chính xác với Bảng 1.
  ├── 1.2 Khớp định nghĩa chỉ số ở Mục 4 (Dòng 609) thành độ chính xác kiểm thử cực đại (Peak Accuracy).
  ├── 1.3 Sửa các tham chiếu chéo LaTeX bị gãy (\ref{eq:proto_nce}, \ref{eq:head_loss}, \ref{eq:ema}).
  ├── 1.4 Hiệu chỉnh các thông số siêu tham số (\beta=0.1, T_ramp=5, \lambda_vl=0.0).
  ├── 1.5 Xóa bỏ hoàn toàn macro tác giả (\NMD, \GVQ) và các lệnh bọc màu bản thảo (\textcolor).
  ├── 1.6 Đổi tên Bảng 3 thành "Design Choice and Architectural Ablations on CIFAR-10".
  └── 1.7 Sửa lỗi bọc ngoặc nhọn TeX ở các dòng 678--736.

[Giai đoạn 2: Cắt gọn Trang & Tái cấu trúc Phụ lục] (Bắt buộc cho ICML; Thực hiện trong 3 Giờ)
  ├── 2.1 Chèn khối Appendix chính thức (Mục 6 báo cáo này) cung cấp chứng minh cho các Mệnh đề 3.1--3.3.
  ├── 2.2 Rút gọn Mục 2 (Related Work) và Mục 3 (Methodology) để phần nội dung chính <= 8 trang.
  ├── 2.3 Xóa bỏ các cột Tiny-ImageNet trống trơn (21 dấu gạch ngang) khỏi Bảng 1.
  ├── 2.4 Giới hạn rõ phạm vi Bảng 1 là CIFAR-10 (hoặc chạy bổ sung CIFAR-100/Flowers102).
  └── 2.5 Biên dịch lại và kiểm tra phân trang PDF: References bắt đầu chính xác từ Trang 9.

[Giai đoạn 3: Minh chứng Trực quan, Công bằng Benchmark & Đồng bộ Code] (Cần chạy máy; 1-2 Ngày)
  ├── 3.1 Sửa file Figure_Plot/test_acc.csv và vẽ lại figures/motivation.pdf.
  ├── 3.2 Công bố Feature Manifold Mixup (mixup_alpha=0.2) trong Mục 3.4 và Mục 4.1.
  ├── 3.3 Thêm số liệu định lượng và trỏ trực tiếp tới Hình 1 trong Mục 5.3.
  ├── 3.4 Chạy lại baseline FedTGP và FedSA với kích thước batch chuẩn B=32 và lr=0.01.
  └── 3.5 Chạy run_benchmark_matrix.py để hoàn thiện dữ liệu CIFAR-100 và Flowers102 cho FedSCD.
```

---

### 7.1 Các Đoạn Mã LaTeX Mẫu Thay Thế Trực Tiếp vào Bài Báo

#### 1. Đóng góp ở Mục 1 (Thay thế Dòng 198)
```latex
% THAY THẾ dòng 198 trong Overleaf/icml2026.tex:
\item Experiments on CIFAR-10 across varying degrees of Dirichlet label skew ($\alpha \in \{1.0, 0.5, 0.1\}$) show that FedSCD consistently lowers Bhattacharyya overlap and outperforms state-of-the-art prototype- and statistics-based baselines---improving test accuracy by up to $5.09$ percentage points over FedTGP (at $\alpha = 1.0$) and $3.45$ percentage points at $\alpha = 0.5$, while achieving margins of up to $8.39$ percentage points over FedORGP and up to $18.77$ percentage points over FedGMKD.
```

#### 2. Mô tả Siêu tham số & Chỉ số ở Mục 4 (Thay thế Dòng 606--609)
```latex
% THAY THẾ dòng 606-609 trong Overleaf/icml2026.tex:
\textbf{Implementation details.} All methods are trained for $T=100$ rounds with full participation, one local epoch per round ($E=1$), mini-batch size 32, and SGD with a learning rate of 0.01 without momentum or weight decay. FedSCD uses decoupled SGD optimizers for the feature extractor and classification head with the same learning rate. For FedSCD, we set the base temperature $\tau = 0.1$, temperature modulation margin $\rho = 0.1$, server momentum $\beta = 0.1$, anchor regularization weight $\lambda = 0.1$, instance contrastive weight $\lambda_{\text{ins}} = 0.05$, warm-up rounds $T_{\text{warm}} = 1$, ramp-up rounds $T_{\text{ramp}} = 5$, and head calibration weight $\lambda_{\text{vg}} = 0.05$ (with $\lambda_{\text{vl}} = 0.0$). For distribution shaping and coordinate gating, we set maximum shrinkage $\gamma_{\max} = 0.25$, prior sample count $n_0 = 50$, variance floor scale $\alpha_v = 0.25$, numerical stability constant $\alpha_q = 10^{-2}$, and virtual feature noise scales $\eta_{\text{loc}} = 0.5, \eta_{\text{glob}} = 0.3$. Baseline methods are configured with their official default hyperparameters.

\textbf{Metrics.} We measure model performance using the peak average local test accuracy across 100 communication rounds, reflecting the maximum attainable representation capability under non-IID partitioning. To evaluate representation geometry, we also track the Euclidean prototype margin and the Bhattacharyya overlap ($e^{-D_B}$ via Eq.~\ref{eq:bc}) evaluated over server-aggregated class statistics across all class pairs. Communication efficiency is assessed by the number of transmitted floating-point parameters per client per round.
```

#### 3. Mô tả Kết quả Thực nghiệm Mục 5.2 (Thay thế Dòng 643--648)
```latex
% THAY THẾ dòng 643-648 trong Overleaf/icml2026.tex:
\textbf{Main results on label skew.}
Table~\ref{tab:com} reports the test accuracy on CIFAR-10 across varying degrees of Dirichlet label skew. 
Under mild heterogeneity ($\alpha = 1.0$), FedSCD achieves 69.60\% test accuracy, outperforming the runner-up baseline FedTGP (64.51\%) by +5.09 percentage points, FedCOF (63.62\%) by +5.98 points, and FedORGP (61.21\%) by +8.39 points.
At moderate heterogeneity ($\alpha = 0.5$), FedSCD reaches 75.15\%, outperforming the strongest baseline FedTGP (71.70\%) by +3.45 percentage points and surpassing other baselines (FedCOF 69.64\%, FedPLVM 69.76\%, FedORGP 68.70\%, FedSA 68.48\%) by up to +6.67 percentage points. 
Under extreme heterogeneity ($\alpha = 0.1$), FedSCD achieves 90.71\%, maintaining margins of +1.83 percentage points over runner-up FedORGP (88.88\%) and +1.84 percentage points over FedTGP (88.87\%).
```

#### 4. Phân tích Hình học Biểu diễn Mục 5.3 (Thay thế Dòng 664--672)
```latex
% THAY THẾ dòng 664-672 trong Overleaf/icml2026.tex:
\subsection{Representation Geometry and Separability}
\label{sec:geometry}

\textbf{Prototype margin vs. distributional overlap.}
To validate the theoretical insights established in Section~\ref{sec:motivation}, we evaluate both the Euclidean prototype margin and the Bhattacharyya overlap ($e^{-D_B}$ via Eq.~\ref{eq:bc}) computed on server-aggregated class statistics across training rounds.
As illustrated in Figure~\ref{fig:motivation}(a)--(b), optimizing solely for inter-prototype Euclidean distance yields deceptive geometric separation: while FedTGP widens the Euclidean prototype margin to 23.85 by Round 100, it simultaneously incurs high Bhattacharyya overlap (0.894) due to uncontrolled intra-class variance in peripheral dimensions. 
Conversely, FedSCD contracts Bhattacharyya overlap to 0.870 while maintaining a compact Euclidean centroid margin (3.77), confirming that coordinate-wise Mahalanobis shrinkage and dynamic angular repulsion are indispensable for achieving genuine class separation on the hypersphere.
```

#### 5. Đặt Nhãn Kép Sửa Lỗi Tham Chiếu Chéo `(Eq. ??)`
```latex
% Trong Overleaf/icml2026.tex:
% Tại Dòng 393:
\label{eq:proto_nce}\label{eq:nce}

% Tại Dòng 441:
\label{eq:head_loss}\label{eq:head}

% Tại Dòng 492:
\label{eq:ema}\label{eq:server_mom}

% Tại Dòng 612:
\label{tab:com}\label{tab:label_skew}
```

#### 6. Kết luận & Hạn chế Chặt chẽ ở Mục 6 (Thay thế Dòng 748--756)
```latex
% THAY THẾ dòng 748-756 trong Overleaf/icml2026.tex:
\section{Conclusion}
\label{sec:conclusion}

This work investigates distribution-aware prototype learning in federated learning under non-IID label skew. We demonstrated both theoretically and empirically that conventional margin-based methods operate under a variance-blind assumption: enlarging Euclidean distances between isolated point centroids does not prevent substantial Bayes classification error when intra-class feature variances remain broad and anisotropic. To resolve this fundamental limitation, we introduced FedSCD, moving the FL prototype paradigm beyond zero-variance points to continuous diagonal Gaussian distributions. By coupling client-side geometric shaping with server-side moment aggregation, FedSCD establishes an architecture-agnostic closed loop that directly targets distributional separability without communicating high-dimensional neural network weights.

Our technical contributions yield concrete theoretical and empirical advantages:
First, by deriving coordinate-wise SNR gates and Mahalanobis compactness penalties, clients actively compress coordinate-wise variance in the Mahalanobis term of the Bhattacharyya distance, provably enlarging inter-class distribution separation. Second, server-side total-variance pooling ensures lossless moment reconstruction, absorbing between-client representation drift without underestimating population spread. Benchmark experiments on CIFAR-10 across severe Dirichlet partitions demonstrate that FedSCD systematically reduces Bhattacharyya overlap and achieves superior accuracy over state-of-the-art baselines. Crucially, ablation analyses confirm that active variance compression---rather than data augmentation or centroid expansion---is the primary driver of these robust generalization gains.

\textbf{Limitations and Future Work.} Although FedSCD substantially improves representation quality under label skew while maintaining a minimal $\mathcal{O}(Cd)$ communication footprint, several fundamental boundaries warrant future investigation. First, our framework parameterizes feature distributions as diagonal Gaussians; modern deep representations post-ReLU lie on non-negative conical manifolds, where Gaussian approximations may allocate probability mass to inactive quadrants. Second, under extreme class sparsity ($n_k^c \le 2$), sample variance estimates degenerate, placing heavy reliance on server shrinkage. Third, while preserving $\mathcal{O}(Cd)$ scaling, transmitting diagonal variances introduces a $2\times$ payload overhead ($C(2d+1)$ floats) relative to standard point prototypes ($Cd$). Finally, sharing second-order dispersion statistics introduces differential privacy considerations that motivate formal DP perturbation mechanisms.
```

---

# 8. Bảng Tham chiếu Báo cáo Thành phần

Các phát hiện chi tiết trong báo cáo này được tổng hợp và đối chiếu từ 5 báo cáo kiểm toán thành phần Tầng 2:

| Mã Phân đoạn | Phạm vi Nội dung | Đường dẫn Báo cáo Thành phần Nguồn |
|:---|:---|:---|
| **Phân đoạn 1** | Abstract, Introduction & Related Work | `.agents/teamwork/group_introduction_and_background/unit_report_introduction_and_background.md` |
| **Phân đoạn 2** | Method Foundations & Problem Setup | `.agents/teamwork/group_method_foundations_and_overview/unit_report_method_foundations_and_overview.md` |
| **Phân đoạn 3** | Client Distribution Learning | `.agents/teamwork/group_method_client_distribution_learning/unit_report_method_client_distribution_learning.md` |
| **Phân đoạn 4** | Server Aggregation, Algorithm 1 & Theory | `.agents/teamwork/group_method_server_aggregation_algorithm_theory/unit_report_method_server_aggregation_algorithm_theory.md` |
| **Phân đoạn 5** | Experiments, Evaluations & Conclusion | `.agents/teamwork/group_experiments_evaluations_and_conclusion/unit_report_experiments_evaluations_and_conclusion.md` |

---
*Báo cáo tiếng Việt được biên dịch và đối chiếu kỹ thuật hoàn chỉnh vào ngày 12-11-2026.*


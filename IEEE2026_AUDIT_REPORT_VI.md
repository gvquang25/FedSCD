# Báo cáo Kiểm toán Kỹ thuật & Rà soát Toàn diện Bản thảo IEEE: FedSCD

> **Bài báo**: *Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning*  
> **Bản thảo mục tiêu**: Tạp chí IEEE Transactions (`Overleaf/ieee2026.tex`, lớp tài liệu `\documentclass[lettersize,journal]{IEEEtran}`)  
> **Chế độ kiểm toán**: Kiểm toán Độc lập Đa chiều & Xác minh Chéo Toàn diện (Chỉ đọc — **Tuyệt đối chưa chỉnh sửa bản thảo gốc**)  
> **Phạm vi đối chiếu**: Bản thảo chính (`Overleaf/ieee2026.tex`, 1.665 dòng), File macro toán (`Overleaf/math_commands.tex`, 522 dòng), Cơ sở dữ liệu tài liệu tham khảo (`Overleaf/ieee2026.bib`, 452 dòng & `ieee2026.bbl`), Nhật ký biên dịch & hình vẽ (`Overleaf/ieee2026.log`, `missfont.log`, `Overleaf/figures/*.pdf`, `Overleaf/FedSCD_*.md`), Mã nguồn thực thi PyTorch (`system/`), Dữ liệu kết quả & kịch bản vẽ hình (`results/`, `Figure_Plot/`, `ABLATION_RUN_GUIDE.md`, `BENCHMARK_RUN_GUIDE.md`), và Báo cáo kiểm toán phiên bản ICML trước đó (`PAPER_REVIEW_REPORT_VI.md`).  
> **Tổng số lỗi độc nhất đã xác minh**: **70 lỗi** (`9 Critical`, `33 Major`, `21 Minor`, `7 Nitpick` — đã loại bỏ hoàn toàn dương tính giả và hợp nhất triệt để mọi phát hiện trùng lặp liên miền).

---

## 1. Tổng quan Kiểm toán & Thống kê Lỗi

### 1.1 Hồ sơ Bản thảo & Các Điểm sáng Kỹ thuật Đã Được Kiểm chứng
Bản thảo `Overleaf/ieee2026.tex` trình bày khung học liên kết **FedSCD** (*Federated Separable Class Distributions*), giải quyết bài toán lệch phân phối nhãn (non-IID label skew) bằng cách mô hình hóa mỗi lớp ngữ nghĩa dưới dạng xấp xỉ Gauss đường chéo $\mathcal{N}(\mu^c, \mathrm{diag}((\sigma^c)^{\odot 2}))$ thay vì một điểm tâm cụm đơn lẻ, kết hợp học tương phản góc có trọng số độ phân tán, chính quy hóa độ nén Mahalanobis có cổng tín hiệu (signal-gated compactness), hiệu chuẩn đầu phân loại tách rời bằng đặc trưng ảo Gauss, và gom moment phía server theo định lý phương sai toàn phần với chi phí truyền thông $\mathcal{O}(Cd)$.

Qua quá trình kiểm toán độc lập trên từng dòng văn bản, công thức và mã nguồn, các khía cạnh nền tảng sau đây của bản thảo `ieee2026.tex` **đã được xác minh là đúng đắn và tiến bộ vượt bậc** so với bản nháp ICML trước đó:
1. **Tính nhất quán hình học giữa không gian Euclid và siêu cầu**: Thống kê Gauss đường chéo $(\mu^c,\sigma^c)$, hàm độ nén Mahalanobis $d^c(z)$ và đặc trưng ảo hoạt động nhất quán trên không gian đặc trưng chưa chuẩn hóa; chỉ hai hàm mục tiêu góc ($\mathcal{L}_{\mathrm{NCE}}$, $\mathcal{L}_{\mathrm{SC}}$) hoạt động trên không gian đã dịch tâm, cân trọng số độ phân tán và chuẩn hóa $\ell_2$ $(\hat{z}, q^c)$ — không xảy ra xung đột hình học giữa giả thiết Gauss đường chéo và chuẩn hóa mặt cầu.
2. **Tính đúng đắn đại số của 3 Mệnh đề Lý thuyết (Propositions III.1–III.3)**: Khác với bản ICML trước đây bị thiếu toàn bộ chứng minh, Mục III-G (`L1373–L1434`) đã bổ sung đầy đủ chứng minh nội dòng (inline proofs) chính xác về mặt đại số cho đồng nhất thức phương sai toàn phần (khi chia cho $n_k^c$ trước bước làm trơn/cắt sàn), tính không âm và điều kiện bằng không khi và chỉ khi của thành phần lệch tâm cụm liên client $B_t^c$, cũng như cận trên Kailath–Bhattacharyya cho sai số phân loại Bayes nhị phân.
3. **Độ phức tạp tính toán/truyền thông & Độ chính xác số học nội bảng**: Các cận tiệm cận $\mathcal{O}(Cd)$, $\mathcal{O}((B+C)^2 d)$, $\mathcal{O}((B+C)Cd)$, $\mathcal{O}(|\mathcal{S}_t|Cd)$ (`L1336–L1364`), toàn bộ 84 ô dữ liệu được điền kín trong `Table I` (không còn ô khuyết `--`), các đánh dấu in đậm/gạch chân (best/second-best), và mọi phép trừ chênh lệch điểm phần trăm ($\Delta$) trong Abstract, Section V và Conclusion đều khớp chính xác đến 2 chữ số thập phân.

---

### 1.2 Thống kê Số lượng Lỗi theo Nhóm Chuyên môn & Mức độ Nghiêm trọng

Toàn bộ kết quả kiểm toán từ 4 miền chuyên môn sau khi xác minh chéo, loại bỏ dương tính giả và khử trùng lặp tuyệt đối thu được đúng **70 lỗi độc nhất**:

| Nhóm Chuyên môn Kiểm toán | Critical (Chí mạng) | Major (Lớn) | Minor (Nhỏ) | Nitpick (Tinh chỉnh) | Tổng cộng | Mã định danh Lỗi |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Phần A** — Toán học, Ký hiệu, Công thức & Chứng minh Lý thuyết | 0 | 5 | 3 | 1 | **9** | `MATH-01`..`MATH-09` |
| **Phần B** — Thuật toán (Pseudocode), Mã nguồn (`system/`) & Thực nghiệm (`results/`) | 4 | 9 | 3 | 0 | **16** | `ALGO-01`..`05`, `CODE-01`..`03`, `CODE-05`..`06`, `EXP-01`..`06` |
| **Phần C** — Chuẩn IEEE (`IEEEtran`), Biên dịch LaTeX, BibTeX (`ieee2026.bib`) & Hình/Bảng | 5 | 12 | 9 | 2 | **28** | `IEEE-01`..`07`, `LATEX-01`..`07`, `FIG-01`..`05`, `BIB-01`..`08`, `VER-01` |
| **Phần D** — Diễn đạt Tiếng Anh Học thuật, Ngữ pháp, Thuật ngữ & Mạch Logic | 0 | 7 | 6 | 4 | **17** | `PROSE-01`..`02`, `04`, `06`..`08`, `STYLE-06`, `TERM-01`..`05`, `LOGIC-02`..`06` |
| **Tổng cộng Toàn Bản thảo** | **9** | **33** | **21** | **7** | **70** | **100% đã xác minh chéo** |

*(Bảng checklist chi tiết theo dõi trạng thái xử lý của từng lỗi từ lỗi thứ 1 đến lỗi thứ 70 được trình bày ngay tại **Mục 1.4** dưới đây.)*

---

### 1.3 Đánh giá Tổng thể Mức độ Sẵn sàng Nộp bài Chuẩn IEEE Transactions

> **Kết luận Kiểm toán**: Bản thảo `Overleaf/ieee2026.tex` ở trạng thái hiện tại **CHƯA SẴN SÀNG ĐỂ NỘP (Not Ready for Submission)** cho bất kỳ tạp chí IEEE Transactions nào. Nếu nộp nguyên trạng, bản thảo chắc chắn sẽ bị **từ chối ngay vòng kiểm tra hình thức của tòa soạn (Administrative Desk-Reject)**, và ngay cả khi qua vòng sơ loại cũng đối mặt nguy cơ bị **Reject / Major Revision nặng** từ hội đồng phản biện kỹ thuật.

Cụ thể, có **4 rào cản chặn đứng việc nộp bài** cần xử lý triệt để trước tiên:
1. **Vi phạm hình thức gây Desk-Reject tức thì tại hệ thống nộp bài IEEE (`IEEE-01`, `IEEE-02`)**: Khối tác giả, đơn vị công tác và tiêu đề chạy đầu trang (`L67–L73`) vẫn đang để nguyên văn mẫu giả `Firstname1 Lastname`, `Department of XXX, University of YYY`, `IEEE Transactions on ...`; đồng thời 7 tiêu đề mục in nhãn ngày nháp (`-New-12/11`, `New 13/11`), 5 ghi chú nội bộ màu xanh đậm `[NMD: ...]` hiển thị trực tiếp trong thân bài/caption Bảng III, và ~35 đoạn văn in màu đỏ rực xuyên suốt Mục III và V.
2. **Vi phạm tính liêm chính học thuật trong Danh mục Tài liệu Tham khảo (`BIB-01`, `BIB-02`)**: Tài liệu `[10]` (`chen2025hproto`, được trích dẫn tại `L115` dưới tên tạp chí *IEEE Transactions on Mobile Computing*, 2025) là **bài báo không tồn tại / bịa đặt (hallucinated reference)** cùng với `liu2025fedpeft` trong file `.bib`; tài liệu `[21]` (`oh2022fedbabu`, ICLR 2022) bị **sai tên bài báo và bịa thêm 4 đồng tác giả không có thật**.
3. **Bất đối xứng giao thức thực nghiệm, báo cáo số liệu đỉnh & lệch cấu hình nghiêm trọng giữa Bài báo và Mã nguồn (`CODE-01`, `CODE-02`, `EXP-01`, `EXP-02`)**:
   - `FedSCD` chạy **101 vòng huấn luyện** và đánh giá *sau* khi train (khiến tại `Round 0` trên Hình 1 và Hình 3, `FedSCD` đã train xong 1 vòng đạt `41%–49%`, trong khi cả 6 baseline đánh giá trên trọng số ngẫu nhiên `~8%–10%` và chỉ train tối đa **100 vòng**).
   - Phép tăng cường ảnh lật ngang + cắt đệm phản xạ (`_apply_scd_view`) **chỉ được cài đặt riêng trong `clientSCD`**, hoàn toàn vắng mặt ở cả 6 baseline — dẫn đến nghịch lý dòng `Baseline (CE + Augmentation only)` không hề giao tiếp liên kết trong `Table II` (`72.79%`) lại cao hơn tất cả 6 thuật toán FL trong `Table I` (`FedTGP` đạt `71.70%`).
   - 5 siêu tham số trọng yếu trong bài báo (`L1453`: $\beta=0.6, \lambda=0.3, \lambda_{\mathrm{ins}}=0.75, \lambda_{\mathrm{vl}}=0.3, \lambda_{\mathrm{vg}}=0.3$) lệch từ **$3\times$ đến $15\times$** (hoặc bật so với tắt $\lambda_{\mathrm{vl}}=0.0$) so với toàn bộ mã nguồn và script chạy thực nghiệm trong repo.
   - `Table I` âm thầm báo cáo `max_acc` (độ chính xác đỉnh), che khuất việc `FedTGP` sụp đổ xuống `0.54%–0.58%` trên Flowers102 và `10/12` lượt chạy `FedPLVM` bị crash giữa chừng, đồng thời văn bản Mục V-A hai lần dẫn chiếu độc giả tới *"limitations in Section IV"* hoàn toàn không tồn tại trong Mục IV.
4. **Sai sót công thức & nhân quả thời gian ngay trên Sơ đồ Kiến trúc chính `Figure 2` (`FIG-01`)**: File `Overleaf/figures/framework.pdf` viết sai tham số phân phối Gauss thành độ lệch chuẩn $\mathcal{N}(p^c, \tilde{\sigma}^c)$, viết công thức EMA vô nghĩa toán học `$(1-\beta)_{t-1} + (\beta)_t$`, dùng nhiệt độ tương lai $\phi_t^c$ (thay vì $\phi_{t-1}^c$) ở phía client vòng $t$, và vẽ thiếu nhánh lan truyền thuận không chặn gradient từ $z$ qua $h_k$ cho $\ell_{\mathrm{CE}}(h_k(z), y)$.

---

### 1.4 Bảng Checklist Tổng hợp Toàn bộ 70 Lỗi Đã Xác minh

> **Hướng dẫn sử dụng**: Tác giả có thể đánh dấu `[x]` trực tiếp vào cột `Trạng thái` khi hoàn thành sửa từng mục. Cột `Giai đoạn` đối chiếu trực tiếp với lộ trình ưu tiên ở **Mục 4** (`GĐ 1`: 9 lỗi `Critical` chặn Desk-Reject & sai lệch cốt lõi; `GĐ 2`: 33 lỗi `Major` hoàn thiện toán học, code, bảng biểu & cấu trúc `IEEEtran`; `GĐ 3`: 21 lỗi `Minor` + 7 lỗi `Nitpick` tinh chỉnh xuất bản). Phân tích kỹ thuật và đoạn code/câu văn sửa mẫu của từng mã lỗi nằm đầy đủ tại **Mục 3**.

| Trạng thái | Mã lỗi | Nhóm | Mức độ | Giai đoạn | Vị trí chính | Tóm tắt lỗi & Hành động cần sửa |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| [ ] | `MATH-01` | Phần A | `Major` | GĐ 2 | `ieee2026.tex` `L827–L912`, `L1175–L1256`, `L1452–L1456` | Bỏ quên giá trị số của 7 hằng số đã hứa ở Mục III-D (gồm siêu tham số cấu trúc $\kappa=10$) và thiếu ràng buộc miền dương $\rightarrow$ Bổ sung $\kappa=10, \epsilon_\mu=\epsilon_g=\epsilon_v=10^{-8}, \epsilon_\sigma=10^{-4}, \epsilon_s=\epsilon_r=10^{-6}, \epsilon_{\mathrm{norm}}=10^{-12}$ vào Mục IV và điều kiện miền dương ở Mục III-D/E. |
| [ ] | `MATH-02` | Phần A | `Major` | GĐ 2 | `ieee2026.tex` `L608–L636`, `L1337–L1388`, `L1455` | Mâu thuẫn cho lớp đơn mẫu ($n_k^c=1$: `Eq. (2)` cho $\sigma_k^c=\mathbf{0}$ nhưng Mục IV gán $10^{-6}\mathbf{1}$) và thiếu quy ước đệm $\mathbf{0}$ khi $n_k^c=0$ $\rightarrow$ Tách phương sai thực nghiệm $(s_k^c)^{\odot 2}$ với $\sigma_k^c$ tải lên có fallback $10^{-6}\mathbf{1}$ ($n_k^c=1$) và zero-padding ($n_k^c=0$). |
| [ ] | `MATH-03` | Phần A | `Major` | GĐ 2 | `ieee2026.tex` `L951–L978` (`Eq. 13`) | Tập hợp vector làm sập bội số mẫu trùng và đụng độ 3 ký hiệu ($p, n, j$) trong $\mathcal{L}_{\mathrm{SC}}$ $\rightarrow$ Định nghĩa trên họ chỉ số hợp rời $\mathcal{R}_i = (\mathcal{I}\setminus\{i\})\sqcup\mathcal{A}$ với biến chạy chỉ số $r, s$ và chỉ số mẫu $\ell$. |
| [ ] | `MATH-04` | Phần A | `Major` | GĐ 2 | `ieee2026.tex` `L797`, `L910–L911`, `L1039–L1110` (`Eqs. 16–17`) | Chưa định nghĩa chỉ số $i$ trên $\widetilde{z}_i^{y_i,\mathrm{loc}}$, đụng độ ký hiệu nhiễu $\epsilon$, mẫu số `Eq. (17)` chia $0$ khi $\mathcal{I},\mathcal{A}=\emptyset$, và $\lambda_{k,t}$ chưa đóng kín $\rightarrow$ Đổi sang $\epsilon_{\mathrm{norm}}$ và $\zeta_i^{\mathrm{loc}}, \zeta^{c,\mathrm{glob}}$, thêm hàm chỉ báo $\mathbb{I}_{k,t}^{\mathrm{stat}}$ và dùng $\operatorname{clip}(\cdot,0,1)$ cho $\lambda_{k,t}$. |
| [ ] | `MATH-05` | Phần A | `Major` | GĐ 2 | `ieee2026.tex` `L1464`, `L1608`, `L1620`; `motivation_metrics.py` | Chưa viết tường minh công thức chồng lấn thực nghiệm $D_B^{\mathrm{eval}}(c,c')$, cách chọn tâm $a^c$, và biến thể `"Class-specific projection"` $\rightarrow$ Bổ sung công thức chính xác khớp `motivation_metrics.py` ($\tilde{v}_{t,j}^c = \max(\bar{v}_{t,j}^c, 10^{-4}) + 10^{-6}$) vào Mục IV và công thức $w^{y_i}, w^c$ vào Mục V-C. |
| [ ] | `MATH-06` | Phần A | `Minor` | GĐ 3 | `ieee2026.tex` `L670–L683`, `L908`, `L934–L1006`, `L1176`, `L1396` | Đụng độ ký hiệu nội phương trình & liên mục ($d^c(z)$ vs. số chiều $d$, $z_j$ mẫu vs. tọa độ, $m$ vs. $m_j^c$, $\bar{\sigma}_j^2$ vs. $\bar{\sigma}_t^c$, $B_t^c$ vs. $B$) $\rightarrow$ Đổi hàm độ nén thành $\psi^c(z)$, tọa độ $[z]_j$, kỳ vọng $\nu_{k,j}^c$ và $\bar{\sigma}_{c,c',j}^2$. |
| [ ] | `MATH-07` | Phần A | `Minor` | GĐ 3 | `ieee2026.tex` `L611–L623`, `L823–L857`, `L1144–L1165`, `L1454` | Dùng chung ký hiệu $(\mu_k^c, n_k^c)$ cho cả cache đầu vòng và thống kê tính mới cuối vòng, đồng thời ẩn chỉ số client $k$ không thông báo $\rightarrow$ Ký hiệu cache vòng trước là $(\breve{\mu}_k^c, \breve{n}_k^c)$ và khai báo việc ẩn chỉ số $k$ ở đầu Mục III-D. |
| [ ] | `MATH-08` | Phần A | `Minor` | GĐ 3 | `ieee2026.tex` `L797`, `L833–L906`, `L1084`, `L1176–L1180` | Cộng trực tiếp vô hướng với vector (`0.5 + 0.5s^c` thiếu $\mathbf{1}$), dùng chung gạch phân số cho chia Hadamard, thiếu khai báo `max` từng tọa độ và viết $\mathcal{N}(0,\mathbf{I})$ số `0` thường $\rightarrow$ Dùng $\mathbf{1}$, toán tử $\oslash$, khai báo $\max$ theo tọa độ và chuẩn hóa $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$. |
| [ ] | `MATH-09` | Phần A | `Nitpick` | GĐ 3 | `ieee2026.tex` `L676–L701`, `L942–L1247`, `L1379–L1430` | Trộn `\ln` và `\log`, thiếu điều kiện dương $\pi_c,\pi_{c'}>0$ ở `L697`, lược bỏ miền tích phân/tổng, viết $\mathcal{L}_{\mathrm{mix}}$ bằng lời và chưa định nghĩa $\operatorname{sg}, \operatorname{clip}, \operatorname{round}$ $\rightarrow$ Đồng bộ `\ln`, thêm `positive` ở `L697`, ghi rõ cận và định nghĩa ngắn gọn các toán tử. |
| [ ] | `ALGO-01` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L586–L601`, `L1273–L1284`, `L1316–L1318` (`Alg. 1`) | `Algorithm 1` thiếu `\ENSURE`, thiếu $\{\mathcal{D}_k\}$, thiếu khởi tạo trọng số chung $(\theta_f^{(0)}, \theta_h^{(0)})$ và quên cập nhật cache cục bộ $\mu_k^c$ $\rightarrow$ Bổ sung đầy đủ `\REQUIRE`/`\ENSURE`, bước khởi tạo đồng nhất và lưu cache $(\breve{n}_k^c, \breve{\mu}_k^c)$. |
| [ ] | `ALGO-02` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L913–L923`, `L1301–L1315`, `L1351–L1353` (`Alg. 1`) | Đặt phép chiếu mẫu $\hat{z}$ ra ngoài vòng lặp minibatch, chưa ghi phạm vi đạo hàm $\nabla_{\theta_{h_k}}$ vs. $\nabla_{\theta_{f_k}}$, và đếm thiếu lượt truyền thuận thứ hai qua $h_k(z_i)$ ở mục Complexity $\rightarrow$ Đưa trích xuất $z_i$ & chiếu $\hat{z}_i$ vào trong minibatch, ghi rõ biến đạo hàm và bổ sung lượt $h_k(z_i)$ trong Complexity. |
| [ ] | `ALGO-03` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L1108–L1109`, `L1294–L1312`; `clientscd.py:412` | Khớp cứng điều kiện $\lambda > 0$ làm tắt nhầm đặc trưng ảo của head khi $\lambda=0$, và gọi vô điều kiện `Eqs. (16)–(17)` ở vòng khởi động lạnh gây $0/0$ $\rightarrow$ Tách cổng $\mathbb{I}_{k,t}^{\mathrm{stat}}$ khỏi $\lambda > 0$ (hoặc ghi rõ ở Mục III-D) và bảo vệ nhánh cold-start. |
| [ ] | `ALGO-04` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L1320–L1325` (`Alg. 1`); `serverscd.py:190` | Bước 19 của `Algorithm 1` gom moment server mà thiếu điều kiện lọc lớp hoạt động $\bar{n}_t^c > 0$ (gây chia $0/0$ khi lớp vắng mặt) $\rightarrow$ Thêm điều kiện `For each class c with \bar{n}_t^c > 0` trước khi gọi `Eqs. (18)–(19)`. |
| [ ] | `ALGO-05` | Phần B | `Minor` | GĐ 3 | `ieee2026.tex` `L472`, `L768–L770`, `L1260`, `L1330–L1341`, `L1464`, `L1535` | Mâu thuẫn dung lượng gói phát sóng downlink giữa bộ 4 có $n_t^c$ ($C(2d+2)$, client không dùng `count`) và bộ 3 không có $n_t^c$ ($C(2d+1)$) $\rightarrow$ Bỏ $n_t^c$ dư khỏi downlink để đối xứng $C(2d+1)$ hoặc thống nhất mô tả bộ 4 trên toàn bài. |
| [ ] | `CODE-01` | Phần B | `Critical` | GĐ 1 | `ieee2026.tex` `L199–L209`, `L1285`, `L1453`, `L1516–L1532`; `serverscd.py:48–77` | Vòng lặp `range(global_rounds + 1)` làm `FedSCD` train **101 vòng** và `evaluate()` *sau* khi train (6 baseline train 100 vòng và `evaluate()` *trước*), làm lệch ngang 1 vòng trên trục `0..100` của `Fig. 1` & `Fig. 3` $\rightarrow$ Đồng bộ vòng lặp `serverscd.py` về đúng 100 vòng pre-train eval hoặc công bố rõ & hiệu chỉnh trục hoành. |
| [ ] | `CODE-02` | Phần B | `Critical` | GĐ 1 | `ieee2026.tex` `L1453`, `L1624`; `main.py`, `clientscd.py`, `run_target_experiments.py` | 5 siêu tham số cốt lõi ($\beta=0.6, \lambda=0.3, \lambda_{\mathrm{ins}}=0.75, \lambda_{\mathrm{vl}}=0.3, \lambda_{\mathrm{vg}}=0.3$) lệch $3\times$–$15\times$ (hoặc bật vs. tắt) so với toàn bộ code & script (chạy script hiện tại ra `74.61%`, thua `FedCOF` ở 6/12 cấu hình) $\rightarrow$ Xác minh log chạy `76.66%` và đồng bộ 100% giữa bài báo, code và script/guides. |
| [ ] | `CODE-03` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L844–L848`, `L1453–L1454`; `clientscd.py:185` | Công thức $n_0 = \max(5, \lvert\mathcal{D}_k\rvert/(2C))$ trong bài báo mâu thuẫn với hằng số cứng `50.0` trong `clientscd.py:185` $\rightarrow$ Sửa `L1454` thành $n_0 = 50$ (hoặc cập nhật code và thêm chỉ số $n_{0,k}$ ở `Eq. (5)` nếu thực tế đã đổi công thức). |
| [ ] | `CODE-05` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L578`, `L1078–L1088`, `L1443–L1456`; `models.py`, `clientbase.py` | Bỏ sót `ReLU` sau `fc1` ($z \in \mathbb{R}_{\ge 0}^d$), `DataLoader(shuffle=False, drop_last=True)` làm rơi vĩnh viễn mẫu dư, crop chung toàn batch và cách gộp/resize ảnh $64\times 64$ $\rightarrow$ Mô tả đầy đủ kiến trúc CNN có `ReLU` cuối, tiền xử lý tập dữ liệu và cấu hình `DataLoader` ở Mục IV. |
| [ ] | `CODE-06` | Phần B | `Minor` | GĐ 3 | `clientscd.py:41–45`; `motivation_metrics.py:18–67`; `ieee2026.tex` `L1464`, `L1550` | Bẫy cờ CLI `-lp 1.0` bị bỏ qua trong `clientscd.py` và khác biệt chế độ trích xuất thống kê vẽ hình (`FedSA` gom online ở `train()` mode vs. `drop_last=False`) $\rightarrow$ Chú thích nguồn thống kê ở Mục V-B và đặt mặc định `--lamda_proto` là `None` trong `main.py`. |
| [ ] | `EXP-01` | Phần B | `Critical` | GĐ 1 | `ieee2026.tex` `L1441–L1532`, `L1641`, `L1657`; `wandb_seed2006_max_acc.csv` | Không công bố quy tắc lấy đỉnh `max_acc` ở `Table I`, dẫn chiếu khống 2 lần tới *"Section IV limitations"* không tồn tại, và che khuất việc `FedTGP` sụp đổ cuối kỳ (`0.54%`) / `10/12` run `FedPLVM` bị crash $\rightarrow$ Khai báo rõ quy tắc chọn checkpoint (kèm số vòng cuối & chú thích run lỗi) ở Mục IV và xóa các tham chiếu treo. |
| [ ] | `EXP-02` | Phần B | `Critical` | GĐ 1 | `ieee2026.tex` `L1455–L1456`, `Table I–II`; `clientscd.py:70–84` & 6 baseline clients | Augmentation ảnh GPU (`_apply_scd_view`) và head mixup chỉ bật ở `FedSCD` (6 baseline tắt hết), gây nghịch lý dòng cục bộ không FL ở `Table II` (`72.79%`) thắng toàn bộ 6 baseline FL ở `Table I` ($\le 71.70\%$) $\rightarrow$ Bật chung augmentation cho cả 6 baseline và chạy lại `Table I` (hoặc công bố thẳng thắn & thêm ablation `w/o Augmentation`). |
| [ ] | `EXP-03` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L1570–L1591`, `L1626–L1640`; `ABLATION_RUN_GUIDE.md:107` | Đặt tên sai dòng `"Baseline (CE + Augmentation only)"` ở `Table II` (ẩn việc ngắt giao tiếp FL + vẫn bật Head Mixup $\rho_{\mathrm{mix}}=0.25$) và thiếu ablation cho $B_t^c=0, \gamma_{\max}=0, g^c=\mathbf{1}$ $\rightarrow$ Đổi tên đúng bản chất `Local-Only` và bổ sung các dòng ablation cốt lõi còn thiếu. |
| [ ] | `EXP-04` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L1463–L1490`, `L1562–L1617` (`Tables I–III`) | Toàn bộ bảng chỉ chạy 1 seed (`seed=2006`) thiếu `mean ± std`, và `Table III` đặt tên *"Sensitivity to ... hyperparameters"* nhưng chỉ bật/tắt 3 công tắc nhị phân $\rightarrow$ Chạy tối thiểu 3 seed, xóa `\NMD` ở `L1595`, bổ sung sweep siêu tham số thực sự hoặc đổi tên/gộp `Table III`. |
| [ ] | `EXP-05` | Phần B | `Minor` | GĐ 3 | `ieee2026.tex` `L1481–L1483` (`Table I`); `wandb_seed2006_max_acc.csv` | 5 ô baseline trong `Table I` lệch làm tròn `0.01–0.02` pp so với dữ liệu CSV gốc $\rightarrow$ Cập nhật chính xác thành `39.98`, `56.38`, `61.20`, `40.86`, `20.34`. |
| [ ] | `EXP-06` | Phần B | `Major` | GĐ 2 | `ieee2026.tex` `L199–L209`, `L716–L728`, `L1449–L1454`, `L1541–L1546` | Không công bố việc `FedPLVM` và `FedGMKD` cũng truyền toàn bộ trọng số CNN $\mathcal{O}(\lvert\theta\rvert)$ mỗi vòng, thiếu siêu tham số của 6 baseline ở Mục IV, và bỏ sót nhận xét hành vi `FedORGP` trên `Fig. 1` $\rightarrow$ Khai báo rõ chi phí truyền backbone + siêu tham số baseline ở Mục IV và bổ sung `FedORGP` cạnh `FedTGP`. |
| [ ] | `IEEE-01` | Phần C | `Critical` | GĐ 1 | `ieee2026.tex` `L54–L60`, `L318`, `L398`, `L441`, `L496`, `L582–L1271`, `L1532`, `L1557`, `L1595` | Cờ ngày nháp (`-New-12/11`, `New 13/11`) trong 7 tiêu đề mục, 5 bình luận nội bộ `[NMD: ...]` màu xanh và ~35 đoạn chữ đỏ `\textcolor{red}{...}` hiển thị trên PDF $\rightarrow$ Xóa sạch nhãn ngày, giải quyết & xóa `\NMD`, gỡ `\textcolor{red}` về chữ đen và xóa macro `L54–L60`. |
| [ ] | `IEEE-02` | Phần C | `Critical` | GĐ 1 | `ieee2026.tex` `L67–L73` (`\author`, `\thanks`, `\markboth`) | Để nguyên thông tin giả của template (`Firstname1 Lastname`, `Department of XXX, University of YYY`, `IEEE Transactions on ...`) gây desk-reject tự động $\rightarrow$ Điền tên tác giả thật, đơn vị công tác, email và tên tạp chí IEEE mục tiêu. |
| [ ] | `IEEE-03` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L838`, `L930`, `L980`, `L1022`, `L1134`, `L1189`, `L1218` vs. `L1264–L1619` | Lệnh `\paragraph{Title.}` tàn dư ICML sinh dấu câu kép `.:`, nhảy cấp bookmark `III-D0a` và lệch kiểu với `\textbf{Heading.}` $\rightarrow$ Đổi sang `\subsubsection{Heading}` (bỏ dấu chấm cuối) thống nhất theo chuẩn `IEEEtran`. |
| [ ] | `IEEE-04` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L42–L51` (ảnh hưởng `L1373, L1390, L1410`) | Tùy chọn `[section]` dưới số mục La Mã sinh nhãn lai `Proposition III.1`–`III.3` phi chuẩn IEEE kèm 6 môi trường khai báo thừa $\rightarrow$ Khai báo gọn `\newtheorem{proposition}{Proposition}` đánh số `1–3` toàn bài. |
| [ ] | `IEEE-05` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L17`, `L20–L28`, `L58–L59`, `L265–L314`; `IEEEtran.cls:2262` | Nạp gói phá bố cục `IEEEtran` (`todonotes`+`\marginparwidth`, `enumitem`, `float`), tắt `balance` và để 5 dòng trống trong `itemize` (`L266, 278, 290, 301, 313`) $\rightarrow$ Gỡ các gói xung đột/thừa, bật `\balance` trang cuối và xóa sạch dòng trống bên trong `itemize`. |
| [ ] | `IEEE-06` | Phần C | `Minor` | GĐ 3 | `ieee2026.tex` `L1`; `ieee2026.log:909–910` | Tùy chọn `lettersize` không được định nghĩa trong `IEEEtran.cls` v1.8b gây cảnh báo `Unused global option(s)` $\rightarrow$ Đổi `L1` thành `\documentclass[letterpaper,journal]{IEEEtran}`. |
| [ ] | `IEEE-07` | Phần C | `Minor` | GĐ 3 | `ieee2026.tex` `L77`, `L194, 199`, `L716, 759, 789`, `L1273`, `L1516–L1593` | Dùng `[t]` thay vì `[!t]` cho 6 float (làm trễ hình/bảng & gây `Underfull \vbox`), viết `Figure~\ref` thay vì `Fig.~\ref` ở đầu câu, và đặt ký hiệu toán trong Abstract $\rightarrow$ Đổi sang `[!t]` và chuẩn hóa toàn bộ thành `Fig.~\ref{...}`. |
| [ ] | `LATEX-01` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L39`; `Overleaf/math_commands.tex` `L3, L20, L49, L124, L136` | Nạp file macro ICLR không dùng đến `math_commands.tex` làm ghi đè hỏng lệnh `\eqref` của `amsmath` (`equation 1`), lỗi `\rvu`/`\rvi` và gọi lại `\usepackage` $\rightarrow$ Xóa dòng `\input{math_commands.tex}` ở `L39`. |
| [ ] | `LATEX-02` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L32–L36`, `L89–L1450`, `L1479–L1484` | Shim giả lập `\citet` ánh xạ thành `[N]` trần gây sai ngữ pháp nếu dùng, và thiếu dấu `~` trước `\cite` trong `Table I` $\rightarrow$ Đổi toàn bộ sang `~\cite{...}` chuẩn IEEE (kể cả trong `Table I`) và xóa khối `L32–L36`. |
| [ ] | `LATEX-03` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L1441` và 10 vị trí `L207, 598, 727, 835, 1088, 1437, 1500–1542` | `\section{Experimental Setup}` thiếu `\label`, dẫn tới 7 chỗ gõ cứng `Section IV` (6 chỗ thiếu `~`) và 3 chỗ trỏ mơ hồ bằng lời $\rightarrow$ Thêm `\label{sec:exp_setup}` tại `L1441` và thay cả 10 chỗ bằng `Section~\ref{sec:exp_setup}`. |
| [ ] | `LATEX-04` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L1490–L1492`, `L1566`, `L1597` | Đặt `\vspace{-10pt}` sau `\end{table*}` làm sập khoảng trắng trước tiêu đề Section V trên trang 10, kèm `\vspace{1.5mm}` thủ công ở `Tables II–III` $\rightarrow$ Xóa `\vspace{-10pt}` (`L1491`) và hai lệnh `\vspace{1.5mm}` (`L1566, L1597`). |
| [ ] | `LATEX-05` | Phần C | `Major` | GĐ 2 | `ieee2026.tex` `L685`, `L797`, `L866–L906`, `L937`, `L1001–L1245`, `L1459–L1463` | Đánh số thừa 5 dòng phụ `(6)–(10)` không nhãn trong khi bỏ trống số cho $\widetilde{z}^{c,b}$ và $\bar{\sigma}_t^c$ (khiến `Fig. 2` `L797` trỏ sai sang `(17)`), kèm dấu `:` trước 7 phương trình $\rightarrow$ Đổi `(6)–(10)` sang `align*`, đánh số/gắn nhãn các phương trình chính và bỏ dấu `:` dẫn nhập. |
| [ ] | `LATEX-06` | Phần C | `Minor` | GĐ 3 | `ieee2026.tex` `L111–L112`, `L153–L154`, `L977`, `L1444–L1456`, `L1535`, `L1576–L1579` | Xuống dòng trước `~\citep` tạo khoảng trắng kép, bọc ngoặc `{=}`/`{+}`/`{-}` làm mất khoảng đệm toán tử (25 lần) và trộn `\text` vs. `\mathrm` ở chỉ số dưới $\rightarrow$ Nối liền `~\cite`, gỡ ngoặc `{=}` và đồng bộ `\mathrm{...}`. |
| [ ] | `LATEX-07` | Phần C | `Minor` | GĐ 3 | `Overleaf/ieee2026.log`, `missfont.log`, `ieee2026 2/3.aux`, `ieee2026.tex:1320–1325` | PDF/log lưu sẵn đã cũ & chưa hội tụ nhãn, bước 19 `Alg. 1` bị `Underfull \hbox (badness 10000)`, thiếu font `pcrr7t` cục bộ và còn 3 file rác `.aux`/`.out` xung đột Google Drive $\rightarrow$ Tách dòng bước 19 `Alg. 1`, xóa file rác và chạy đủ 4 lượt `latexmk -pdf`. |
| [ ] | `FIG-01` | Phần C | `Critical` | GĐ 1 | `Overleaf/figures/framework.pdf` & `ieee2026.tex` `L789–L800` (`Fig. 2`) | Sơ đồ `Fig. 2` sai 5 điểm: viết $\mathcal{N}(p^c, \tilde{\sigma}^c)$ thiếu bình phương phương sai, EMA sai cú pháp `$(1-\beta)_{t-1}+(\beta)_t$`, dùng $\phi_t^c$ thay vì $\phi_{t-1}^c$ ở client, khuyết nhãn $\mu^c$, thiếu nhánh $z \to h_k$ không `sg` $\rightarrow$ Vẽ/xuất lại `framework.pdf` và đổi `\textsc{FedSCD}` thành `FedSCD` ở `L797`. |
| [ ] | `FIG-02` | Phần C | `Minor` | GĐ 3 | `Overleaf/figures/framework.pdf` vs. `Overleaf/FedSCD_*.md` | Lệch hoàn toàn bố cục và tên file tham chiếu giữa `framework.pdf` và 2 file hướng dẫn `.md` nằm trong thư mục `Overleaf/` $\rightarrow$ Di chuyển 2 file `.md` nội bộ ra khỏi gói nguồn nộp bài `Overleaf/`. |
| [ ] | `FIG-03` | Phần C | `Major` | GĐ 2 | `Overleaf/figures/motivation.pdf`, `ieee2026.tex` `L199–L209`, `plot_combined_3panels.py` | `Fig. 1` âm thầm bỏ sót baseline `FedPLVM`, thiếu nhãn trục tung cả 3 panel (tiêu đề Panel (b) lệch định nghĩa rescaled overlap) và thu nhỏ hình 2 cột xuống 50% làm chữ còn `~6.5pt` $\rightarrow$ Vẽ lại ở chuẩn 1 cột font `8–9pt`, thêm nhãn trục tung và bổ sung/chú thích `FedPLVM`. |
| [ ] | `FIG-04` | Phần C | `Minor` | GĐ 3 | `Overleaf/figures/ieee_tsne_cifar10_dir05.pdf` | File hình t-SNE mồ côi nằm thừa trong `Overleaf/figures/` không được tham chiếu (đồng thời thiếu `FedPLVM` và thiếu chú giải màu lớp/dấu sao) $\rightarrow$ Xóa khỏi `Overleaf/figures/` hoặc bổ sung đủ panel + legend nếu đưa lại vào Mục V-B. |
| [ ] | `FIG-05` | Phần C | `Minor` | GĐ 3 | `ieee2026.tex` `L1465–L1490`, `L1562–L1584`, `L1593–L1617` (`Tables I–III`) | Dùng `\resizebox` làm méo font ở `Table I`, lệch chữ số thập phân (`21.8`, `-0.9`), thừa ngoặc `{31.00}`, dùng gạch nối `-` thay vì dấu trừ `$-$`, và lệch kiểu viết hoa caption bảng $\rightarrow$ Dùng `\footnotesize`+`tabular*`, chuẩn hóa 2 chữ số thập phân, chế độ toán `$-$` và Title Case. |
| [x] | `BIB-01` | Phần C | `Critical` | GĐ 1 | `Overleaf/ieee2026.bib` `L400–L406` (`[10]`, trích tại `L115`) & `L393–L398` | Tài liệu `[10]` (`chen2025hproto`, gán cho *IEEE TMC* 2025) và `liu2025fedpeft` (CVPR 2025) là bài báo **không tồn tại / bịa đặt** $\rightarrow$ Xóa cả 2 khỏi `ieee2026.bib` và thay thế trích dẫn tại `L115` bằng công trình có thật đã kiểm chứng. |
| [x] | `BIB-02` | Phần C | `Critical` | GĐ 1 | `Overleaf/ieee2026.bib` `L370–L375` (`[21]`, trích tại `L331`) | Tài liệu `[21]` (`oh2022fedbabu`, ICLR 2022) bị sửa sai tiêu đề gốc và bịa thêm 4 đồng tác giả không có thật $\rightarrow$ Sửa về đúng tiêu đề ICLR 2022 và bộ 3 tác giả thật `Oh, Jaehoon and Kim, Sangmook and Yun, Se-Young`. |
| [x] | `BIB-03` | Phần C | `Major` | GĐ 2 | `Overleaf/ieee2026.bib` `L380–L385` và `L322–L323` | Khối BibTeX xóa dở của `FedGH` thiếu dòng mở đầu `@inproceedings{...` gây dư dấu `}` (kèm sai tên tác giả `Shi, Zhuanqiang`) và dòng `year` bị ngắt đôi ở `L322–L323` $\rightarrow$ Xóa khối thừa `L380–L385` và gộp dòng `year = {2021}`. |
| [x] | `BIB-04` | Phần C | `Major` | GĐ 2 | `Overleaf/ieee2026.bib` `L408, L415, L428–L430` (`[15]–[17]`) | Thiếu ngoặc nhọn `{...}` bảo vệ chữ hoa khiến `IEEEtran.bst` in thường sai thành `Gfpl`, `Fedup`, `Fedcdwa`, `wasserstein` $\rightarrow$ Bọc `{GFPL}`, `{FedUP}`, `{FedCDWA}`, `{Wasserstein}` trong `ieee2026.bib`. |
| [x] | `BIB-05` | Phần C | `Major` | GĐ 2 | `Overleaf/ieee2026.bib` `L23–L31` (`[3]`) và `L407–L412` (`[15]`) | Cắt cụt `and others` sai chuẩn IEEE cho bài 5 tác giả `lu2026gfpl` và ghi thiếu số kỳ kép `number = {1--2}` / danh sách tác giả đầu cho `kairouz2021advances` $\rightarrow$ Điền đủ 5 tác giả cho `lu2026gfpl` và cập nhật `kairouz2021advances`. |
| [x] | `BIB-06` | Phần C | `Minor` | GĐ 3 | `Overleaf/ieee2026.bib` `L7–L13, L144–L150, L238–L261, L340–L346` | Thiếu trường `pages` và/hoặc `volume` ở 5 bài kỷ yếu chính thức (`wang2024fedplvm`, `goswami2025fedcof`, `luo2021ccvr`, `collins2021exploiting`, `mcmahan2017fedavg`) $\rightarrow$ Bổ sung đầy đủ số tập và số trang kỷ yếu đã xác minh. |
| [x] | `BIB-07` | Phần C | `Minor` | GĐ 3 | `Overleaf/ieee2026.bib` `L119–L124, L178–L183, L215–L235, L326–L338, L407–L425` | Trùng lặp `fedup2026` vs. `qi2026fedup`, dư 17 entry không trích dẫn, 2 bản arXiv cũ đã đăng hội nghị (`ACM MM 23`, `CVPR 24`) và dùng `@article` cho `arXiv preprint` $\rightarrow$ Gộp entry trùng, dọn `.bib` và chuyển arXiv sang `@misc`. |
| [x] | `BIB-08` | Phần C | `Nitpick` | GĐ 3 | `Overleaf/ieee2026.bib` `L99, L249, L343, L364, L433–L450` | Chuỗi tên hội nghị ICML/ICLR/IEEE viết theo 3 kiểu khác nhau, chưa dùng tên viết tắt chuẩn IEEE và dư `publisher` $\rightarrow$ Đồng bộ theo `IEEEabrv.bib` và lược bỏ các trường `publisher`/`series` dư thừa. |
| [ ] | `VER-01` | Phần C | `Nitpick` | GĐ 3 | `ieee2026.tex` `L80–L82`, `L749–L750`, `L803–L804`, `L1561`, `L1592` | Từ khóa `IEEEkeywords` chưa xếp theo thứ tự bảng chữ cái ABC và còn sót comment nhắc prompt, tên file code cũ (`clientnew_pcl.py`), comment tiếng Việt trong `.tex` $\rightarrow$ Xếp lại `IEEEkeywords` theo ABC và dọn sạch comment nguồn. |
| [ ] | `PROSE-01` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L76–L78` (Abstract) | Lỗi ngữ pháp–logic `"are highest among the six evaluated baselines"` (hàm ý FedSCD là baseline), thiếu từ nối phản đề, viết `"diagonal second moments"` thiếu chữ *central* và giọng rào đón $\rightarrow$ Viết lại Abstract chuẩn IEEE Transactions. |
| [ ] | `PROSE-02` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L1190` (Section III-E) | Lỗi đánh máy thiếu dấu cách sau dấu phẩy (`fluctuate,altering`) in dính chữ trên PDF $\rightarrow$ Sửa thành `fluctuate across rounds, altering`. |
| [ ] | `PROSE-04` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L780–L782`, `L1207–L1208`, `L1316–L1317`, `L1347–L1348` | Dùng biệt ngữ lập trình PyTorch (`statistics loader`, `The implementation`) và câu cụt `; upload.` trong mục phương pháp & `Algorithm 1` $\rightarrow$ Thay bằng ký hiệu toán học $\mathcal{D}_{k,t}^{\mathrm{stat}}$ và mệnh đề hoàn chỉnh. |
| [ ] | `PROSE-06` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L1499–L1500`, `L1514`, `L1519–L1520`, `L1544`, `L1657` | Cụm từ thiếu tự nhiên (`available accuracy point estimates`), hoán dụ (`Section IV evaluates`), khuyết danh từ (`at the strongest`) và thiếu mạo từ đầu câu (`Evaluation uses`) $\rightarrow$ Viết lại 5 câu theo văn phong học thuật chuẩn. |
| [ ] | `PROSE-07` | Phần D | `Nitpick` | GĐ 3 | `ieee2026.tex` `L207, 260, 360, 475, 1255, 1502–1655` | Cụm tính từ ghép 3–4 gạch nối nặng nề (`test-set-size-weighted`), trộn `percentage points` và `points`, số nhiều `numbers of ...`, nối lặp `and ... and` $\rightarrow$ Rút gọn thành `weighted local test accuracy` và đồng bộ `percentage points`. |
| [ ] | `PROSE-08` | Phần D | `Nitpick` | GĐ 3 | `ieee2026.tex` `L1492` (Section V heading) | Tiêu đề mục dùng danh từ số nhiều phi chuẩn `\section{Experimental Evaluations}` $\rightarrow$ Đổi thành `\section{Experimental Results}` (hoặc `Experimental Evaluation`). |
| [ ] | `STYLE-06` | Phần D | `Nitpick` | GĐ 3 | `ieee2026.tex` `L207` (`Fig. 1` caption) & `L1608` (`Table III`) | Gọi hình con là `Panel (a)` / `panel (b)` thay vì `(a)`, `(b)` theo chuẩn IEEE và viết hoa thừa `(vs. Shared)` $\rightarrow$ Sửa thành `(a)`, `(b)`, `(c)` ở `L207` và `(vs. shared)` ở `L1608`. |
| [ ] | `TERM-01` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L77`, `L207`, `L211–L214`, `L797` | Viết ngược thứ tự giải nghĩa `FedSCD (...)` ở Abstract, không giải nghĩa `FedSCD` trong toàn bộ thân bài, dùng trước khi giới thiệu ở `Fig. 1` (kèm `FedCOF, FedSA` thiếu `\cite`) và trộn 3 kiểu chữ $\rightarrow$ Chuẩn hóa `Federated Separable Class Distributions (FedSCD)` đứng thường. |
| [ ] | `TERM-02` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L797`, `L1189–L1212`, `L1437–L1646`, `L1653` | Dùng viết tắt `EMA` 7 lần không định nghĩa xuyên suốt Mục III–V rồi mới giải nghĩa ở tận phần Kết luận (`L1653`) $\rightarrow$ Định nghĩa `exponential moving average (EMA)` ngay lần đầu tại Mục III-C (`L784`) và dùng `EMA` ở Kết luận. |
| [ ] | `TERM-03` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L77, 307, 508, 574, 1447, 1452, 1577, 1581` | Chưa giải nghĩa lần đầu `non-IID` (Abstract), `ProtoNCE`, `CNN`, `SGD`, `SupCon`, `CE`; gán nhầm nhãn `SupCon` cho $\mathcal{L}_{\mathrm{SC}}$ và mở rộng lại thừa `FL` $\rightarrow$ Giải nghĩa đủ ở lần đầu và đổi nhãn bảng thành `w/o Hybrid Contrastive`. |
| [ ] | `TERM-04` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L241, 291, 511, 838, 965, 1119, 1189, 1453, 1577, 1607–1624` | Tên các module kỹ thuật trôi dạt giữa các mục và nhãn `"No momentum (\beta=1.0)"` ở `Table III` gây hiểu ngược chiều tham số $\beta$ trong `Eq. (20)` $\rightarrow$ Chuẩn hóa bộ thuật ngữ cố định và đổi thành `Without EMA smoothing ($\beta=1.0$)`. |
| [ ] | `TERM-05` | Phần D | `Nitpick` | GĐ 3 | `ieee2026.tex` `L77, 245, 442, 456, 991, 1142, 1379, 1464` | Dao động 4 biến thể từ vựng cho phép gộp có trọng số mẫu (`sample-weighted`, `count-pooled`, ...) và chưa giải thích chữ `PL` trong $\mathcal{L}_{\mathrm{PL}}$ $\rightarrow$ Thống nhất dùng `sample-weighted` và chú thích ký hiệu $\mathcal{L}_{\mathrm{PL}}$ ở `L991`. |
| [ ] | `LOGIC-02` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L77, 452, 597, 745, 1013, 1209, 1237, 1436, 1504–1646` | Chèn dày đặc các câu tự phủ định/miễn trừ kiểm toán cuối hầu hết các đoạn làm đứt gãy mạch lập luận khoa học $\rightarrow$ Lược bỏ các câu rào đón vụn vặt và gom các giới hạn thực chất về một mục `Discussion and Limitations` cuối Section V. |
| [ ] | `LOGIC-03` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L387–L397`, `L442–L478`, `L497–L525` & 5–7 vị trí lặp | Section II (*Related Work*) thuyết minh lại toàn bộ cơ chế FedSCD trước Section III, đồng thời lặp lại lưu ý overlap/moment 5–7 lần $\rightarrow$ Cô đọng phần định vị cuối mỗi tiểu mục Section II còn 4–5 câu đối sánh trực tiếp. |
| [ ] | `LOGIC-04` | Phần D | `Major` | GĐ 2 | `ieee2026.tex` `L241–L300`, `L582–L601`, `L1128–L1171`, `L1446`, `L1634–L1638` | Bỏ lửng câu hỏi vì sao các tọa độ đặc trưng căn chỉnh được khi không truyền trọng số backbone, và thiếu thực nghiệm tách riêng $B_t^c=0$ $\rightarrow$ Giải thích vai trò khởi tạo chung $(\theta_f^{(0)},\theta_h^{(0)})$ + neo tọa độ, và bổ sung ablation $B_t^c=0$. |
| [ ] | `LOGIC-05` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L132–L149`, `L207`, `L647–L659`, `L718–L726`, `L1516–L1532` | Dẫn dắt `FedTGP` như ví dụ chồng lấn cao nhưng `Fig. 1(b)` lại cho thấy `FedTGP` chồng lấn thấp hơn `FedSCD`, và `Fig. 3` chỉ vẽ 1 mình `FedSCD` $\rightarrow$ Dẫn dắt theo 2 thái cực (`FedSA/FedCOF` vs. `FedTGP/FedORGP`) và thêm đường baseline vào `Fig. 3`. |
| [ ] | `LOGIC-06` | Phần D | `Minor` | GĐ 3 | `ieee2026.tex` `L591–L610` (Section III-A) | Nhắc tới `"the gradient-free statistics pass"` và quy tắc loại bỏ minibatch lẻ ở `L591–L598` trước khi giới thiệu bước trích xuất sau huấn luyện ở `L605–L607` $\rightarrow$ Đảo lại trình tự diễn giải tự nhiên ở đầu Section III-A. |

---

## 2. Tóm tắt các Lỗi Nghiêm trọng Nhất (`Critical` & `Major`)

### 2.1 Bảng Điểm nhanh Toàn bộ 9 Lỗi Chí mạng (`Critical`)

| Mã lỗi | Phần | Vị trí chính | Tóm tắt Bản chất Lỗi & Hệ quả |
|:---|:---:|:---|:---|
| **`CODE-01`** | B | `ieee2026.tex` `L199–L209`, `L1285`, `L1453`, `L1516–L1532` vs. `serverscd.py:48–77` | Vòng lặp `range(global_rounds + 1)` làm `FedSCD` train **101 vòng** (baseline train 100 vòng) và gọi `evaluate()` *sau* `client.train()` (baseline gọi *trước*), làm lệch ngang 1 vòng trên toàn bộ trục hoành `0..100` của `Fig. 1` và `Fig. 3`. |
| **`CODE-02`** | B | `ieee2026.tex` `L1453`, `L1624` vs. `main.py`, `clientscd.py`, `run_target_experiments.py` | 5 siêu tham số cốt lõi ($\beta, \lambda, \lambda_{\mathrm{ins}}, \lambda_{\mathrm{vl}}, \lambda_{\mathrm{vg}}$) trong bài báo mâu thuẫn hoàn toàn với mặc định code và toàn bộ script benchmark/ablation (chạy script hiện tại ra `74.61%`, thua `FedCOF` ở 6/12 cấu hình thay vì `76.66%`). |
| **`EXP-01`** | B | `ieee2026.tex` `L1458–L1532`, `L1641` vs. `wandb_seed2006_max_acc.csv` | Không công bố quy tắc lấy đỉnh `max_acc` trong `Table I`; dẫn chiếu khống 2 lần tới *"checkpoint and update-budget limitations in Section IV"* không hề tồn tại; che khuất việc `FedTGP` sụp đổ cuối kỳ và `10/12` lượt chạy `FedPLVM` bị crash/chưa chạy xong. |
| **`EXP-02`** | B | `ieee2026.tex` `L1455–L1456`, `Table I–II` vs. `clientscd.py:70–84` & 6 baseline clients | Augmentation ảnh (flip + reflection-padded crop) và head mixup chỉ bật duy nhất ở `FedSCD`, tắt ở cả 6 baseline, gây nghịch lý mô hình cục bộ không giao tiếp FL ở `Table II` (`72.79%`) thắng toàn bộ 6 baseline FL ở `Table I` (`\le 71.70%`). |
| **`IEEE-01`** | C | `ieee2026.tex` `L54–L60`, `L318`, `L398`, `L441`, `L496`, `L582–L1271`, `L1532`, `L1557`, `L1595` | Cờ nháp `-New-12/11`, `New 13/11` trong 7 tiêu đề mục & PDF bookmark, 5 khối bình luận nội bộ `[NMD: ...]` màu xanh in thẳng ra PDF (cả trong caption Bảng III), và ~35 khối chữ đỏ `\textcolor{red}{...}` chưa dọn. |
| **`IEEE-02`** | C | `ieee2026.tex` `L67–L73` | Giữ nguyên văn thông tin tác giả/đơn vị/tạp chí giả định của template (`Firstname1 Lastname`, `Department of XXX, University of YYY`, `IEEE Transactions on ...`) — lỗi bị loại tự động ngay khi nộp. |
| **`FIG-01`** | C | `Overleaf/figures/framework.pdf` & `ieee2026.tex` `L789–L800` | Sơ đồ tổng quan `Fig. 2` chứa 5 lỗi toán học/kiến trúc trực quan: viết $\mathcal{N}(p^c, \tilde{\sigma}^c)$ thiếu bình phương phương sai, viết EMA sai cú pháp `$(1-\beta)_{t-1} + (\beta)_t$`, dùng nhiệt độ tương lai $\phi_t^c$ ở client, thiếu nhãn $\mu^c$, và thiếu đường truyền $z \to h_k$ không `sg`. |
| **`BIB-01`** | C | `ieee2026.bib` `L400–L406` (`[10]`, trích tại `L115`) & `L393–L398` | Tài liệu tham khảo `[10]` (`chen2025hproto`, gán cho *IEEE TMC* 2025) và entry không trích dẫn `liu2025fedpeft` (CVPR 2025) là các bài báo **hoàn toàn không tồn tại (AI-hallucinated citations)**. |
| **`BIB-02`** | C | `ieee2026.bib` `L370–L375` (`[21]`, trích tại `L331`) | Tài liệu `[21]` (`oh2022fedbabu`, ICLR 2022) bị sửa sai tiêu đề bài báo gốc và bịa đặt 4 đồng tác giả (`Han, Yun, Cho, Kim`) thay cho danh sách tác giả thật (`Jaehoon Oh, Sangmook Kim, Se-Young Yun`). |

### 2.2 Tóm lược Chiến lược 33 Lỗi Lớn (`Major`) theo Cụm Chuyên môn
- **Cụm Toán học & Khả năng tái lập công thức (5 Major: `MATH-01`..`MATH-05`)**: Bỏ quên giá trị số của 7 hằng số đã hứa ở Mục III-D, đặc biệt là siêu tham số cấu trúc $\kappa=10$ trong công thức nhiệt độ `Eq. (21)` (`MATH-01`); mâu thuẫn nội bộ khi lớp đơn mẫu $n_k^c=1$ cho $\sigma_k^c=\mathbf{0}$ ở `Eq. (2)` nhưng lại gán $10^{-6}\mathbf{1}$ ở Mục IV và thiếu quy ước đệm $\mathbf{0}$ khi $n_k^c=0$ (`MATH-02`); định nghĩa tập vector làm sập bội số mẫu trùng nhau kèm đụng độ 3 ký hiệu $p, n, j$ trong $\mathcal{L}_{\mathrm{SC}}$ (`MATH-03`); chỉ số $i$ chưa định nghĩa ở $\widetilde{z}_i^{y_i,\mathrm{loc}}$, đụng độ ký hiệu nhiễu $\epsilon$, mẫu số không có hàm chỉ báo bảo vệ khi $|\mathcal{I}|=0$ hoặc $|\mathcal{A}|=0$ ở `Eq. (17)`, và công thức $\lambda_{k,t}$ âm khi chưa qua warm-up (`MATH-04`); thiếu công thức tường minh cho chỉ số chồng lấn thực nghiệm $D_B^{\mathrm{eval}}(c,c')$ cũng như biến thể `"Class-specific projection"` trong Bảng III (`MATH-05`).
- **Cụm Thuật toán, Đối chiếu Code & Thực nghiệm (9 Major: `ALGO-01`..`04`, `CODE-03`, `CODE-05`, `EXP-03`, `EXP-04`, `EXP-06`)**: `Algorithm 1` thiếu `\ENSURE`, thiếu tập dữ liệu cục bộ, thiếu khởi tạo trọng số chung $(\theta_f^{(0)}, \theta_h^{(0)})$ và bước lưu cache $\mu_k^c$ cục bộ (`ALGO-01`); đặt phép chiếu mẫu $\hat{z}$ ra ngoài vòng lặp minibatch, không ghi rõ phạm vi đạo hàm riêng $\nabla_{\theta_{h_k}}$ vs. $\nabla_{\theta_{f_k}}$ và đếm thiếu lượt truyền thuận thứ hai qua $h_k(z_i)$ trong mục Complexity (`ALGO-02`); gộp nhầm điều kiện $\lambda > 0$ làm tắt luôn đặc trưng ảo của head và gọi vô điều kiện `Eqs. (16)–(17)` ở vòng khởi động lạnh gây $0/0$ (`ALGO-03`); thiếu điều kiện bảo vệ $\bar{n}_t^c > 0$ ở bước gom moment server trong `Algorithm 1` (`ALGO-04`); công thức $n_0 = \max(5, |\mathcal{D}_k|/(2C))$ trong bài báo mâu thuẫn với hằng số cứng `50.0` trong `clientscd.py:185` (`CODE-03`); bỏ sót hàm kích hoạt `ReLU` sau `fc1` (khiến $z \in \mathbb{R}_{\ge 0}^d$), `DataLoader(shuffle=False, drop_last=True)` làm rơi vĩnh viễn các mẫu dư và cách gộp/resize tập dữ liệu (`CODE-05`); đặt tên sai bản chất hàng cơ sở trong `Table II` và thiếu ablation cho các trụ cột như $B_t^c=0$ hay $\gamma_{\max}=0$ (`EXP-03`); toàn bộ bảng chỉ chạy 1 seed không có độ lệch chuẩn và `Table III` đặt tên "Sensitivity to hyperparameters" nhưng chỉ bật/tắt 3 công tắc nhị phân (`EXP-04`); không công bố việc `FedPLVM` và `FedGMKD` truyền toàn bộ tham số backbone CNN $\mathcal{O}(|\theta|)$ mỗi vòng, thiếu siêu tham số của các baseline và bỏ sót nhận xét hành vi của `FedORGP` trên `Fig. 1` (`EXP-06`).
- **Cụm Chuẩn `IEEEtran`, LaTeX, Hình vẽ & BibTeX (12 Major: `IEEE-03`..`05`, `LATEX-01`..`05`, `FIG-03`, `BIB-03`..`05`)**: Lệnh `\paragraph{Title.}` tàn dư từ ICML sinh dấu chấm kép `.:` và nhảy cấp tiêu đề `III-D0a` trong `IEEEtran` (`IEEE-03`); đánh số mệnh đề theo mục tạo thành `Proposition III.1` không chuẩn IEEE (`IEEE-04`); nạp các gói phá vỡ bố cục `IEEEtran` (`todonotes` + `\marginparwidth`, `enumitem`, `float`), tắt `balance` và để dòng trống trong `itemize` vi phạm cảnh báo của `IEEEtran.cls` (`IEEE-05`); nạp file thừa `math_commands.tex` ghi đè hỏng lệnh `\eqref` của `amsmath` (`LATEX-01`); shim tương thích `\citet` in ra `[N]` cụt lủn thay vì tên tác giả (`LATEX-02`); thiếu `\label` cho Section IV kèm 10 tham chiếu cứng/mơ hồ (`LATEX-03`); lệnh `\vspace{-10pt}` đặt sau `table*` làm sập khoảng trắng trước tiêu đề Section V trên trang 10 (`LATEX-04`); đánh số tràn lan 5 phương trình trung gian `(6)–(10)` không nhãn trong khi bỏ trống số cho các phương trình cốt lõi và đặt dấu hai chấm trước phương trình hiển thị (`LATEX-05`); `Fig. 1` âm thầm bỏ sót baseline `FedPLVM`, thiếu nhãn trục tung cả 3 panel và co nhỏ canvas 2 cột xuống 50% làm cỡ chữ còn `~6.5pt` (`FIG-03`); mảnh BibTeX cụt đầu thiếu `@inproceedings{...` tại `ieee2026.bib:380–385` (`BIB-03`); thiếu ngoặc nhọn `{...}` làm `IEEEtran.bst` in thường sai tên `Gfpl`, `Fedup`, `Fedcdwa`, `wasserstein` (`BIB-04`); cắt cụt `and others` sai quy định cho bài 5 tác giả `lu2026gfpl` và thiếu số kỳ kép `1--2` cho `kairouz2021advances` (`BIB-05`).
- **Cụm Diễn đạt Học thuật, Thuật ngữ & Mạch Logic (7 Major: `PROSE-01`, `TERM-01`, `TERM-02`, `TERM-04`, `LOGIC-02`..`04`)**: Lỗi ngữ pháp-logic câu so sánh nhất trong Abstract (`"are highest among the six evaluated baselines"`), thiếu từ nối phản đề và giọng văn rào đón kiểu biên bản kiểm toán (`PROSE-01`); viết ngược tên viết tắt `FedSCD` trong Abstract, không bao giờ giải nghĩa `FedSCD` trong toàn bộ thân bài, dùng trước khi giới thiệu ở `Fig. 1` và thiếu đồng bộ kiểu chữ (`TERM-01`); dùng `EMA` 7 lần không định nghĩa ở Mục III–V rồi mới giải nghĩa ở tận Conclusion (`TERM-02`); trôi dạt tên gọi module kỹ thuật giữa các mục và nhãn gây hiểu nhầm `"No momentum (\beta=1.0)"` trong `Table III` (`TERM-04`); giọng điệu tự phủ định/rào đón phòng thủ dày đặc làm đứt gãy mạch văn khoa học (`LOGIC-02`); lặp lại nội dung nặng nề ở Section II (mô tả lại toàn bộ kiến trúc FedSCD trước Section III) và lặp lại cùng một lưu ý 5–7 lần (`LOGIC-03`); bỏ ngỏ lời giải thích vì sao các tọa độ đặc trưng giữa các client lại tương thích ngữ nghĩa khi không chia sẻ trọng số backbone, cũng như thiếu thực nghiệm kiểm chứng riêng thành phần phương sai liên client $B_t^c$ (`LOGIC-04`).

---

## 3. Chi tiết Toàn bộ Lỗi theo Chuyên môn

---

## Phần A — Toán học, Ký hiệu, Công thức & Chứng minh Lý thuyết (9 Lỗi: `MATH-01` – `MATH-09`)

### `MATH-01` — Bảy Hằng số Toán học Được Hứa ở Mục III-D Bị Bỏ quên Giá trị ở Mục IV (Bao gồm Siêu tham số Cấu trúc $\kappa=10$ trong Hiệu chuẩn Nhiệt độ) và Thiếu Ràng buộc Miền Dương
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L827–L836`, `L843–L912` (Section III-D, Eqs. `(5)`–`(11)`), `L1175–L1181` & `L1224–L1256` (Section III-E, Eqs. `(21)`–`(22)`), và `L1452–L1456` (Section IV, *Implementation details*); đối chiếu `system/flcore/clients/clientscd.py` (`L166, L202, L215, L218, L222, L224, L283–L284`) và `system/flcore/servers/serverscd.py` (`L205, L221–L223, L238, L263`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L826-L836:
  \textcolor{red}{Let
  $\mathcal{A}_{t-1}
  =\{c\in\mathcal{C}_{t-1}:
  \|\mu_{t-1}^c\|_2>\epsilon_\mu\}$}
  ...
  Numerical stabilization constants and implementation
  settings are specified in the experimental details.

  % Overleaf/ieee2026.tex, L1224-L1231 (Eq. 21):
  \begin{equation}
  r_t^c
  =
  \frac{\|\sigma_t^c\|_2}
  {(\|\mu_t^c\|_2+\epsilon_r)
   \log(\max(n_t^c,1)+\kappa)},
  \qquad c\in\mathcal{C}_t.
  \label{eq:raw_conc}
  \end{equation}
  ```
- **Phân tích chi tiết**:
  1. **Thất hứa 7 hằng số trong Mục IV**: Dòng `L835–L836` khẳng định rõ rằng mọi hằng số ổn định số học và thiết lập cài đặt đều được nêu trong phần chi tiết thực nghiệm (Section IV). Thế nhưng kiểm tra toàn bộ `L1452–L1456` cho thấy **7 hằng số** xuất hiện trong các phương trình của Section III-D và III-E hoàn toàn không được gán giá trị số:
     - $\kappa$ (`L1229` — thực tế trong `serverscd.py:223, 238` là `10.0`),
     - $\epsilon_\mu$ (`L829, L1289` — thực tế trong `clientscd.py:166` là `1e-8`),
     - $\epsilon_s$ (`L883` — thực tế trong `clientscd.py:215` là `1e-6`),
     - $\epsilon_g$ (`L889, L902, L905` — thực tế trong `clientscd.py:218, 222, 224` là `1e-8`),
     - $\epsilon$ trong $\operatorname{norm}_\epsilon$ (`L910–L911` — thực tế mặc định `F.normalize` tại `clientscd.py:283–284` là `1e-12`),
     - $\epsilon_v$ (`L1179, L1324` — thực tế trong `serverscd.py:205` là `1e-8`),
     - $\epsilon_r$ (`L1228, L1248` — thực tế trong `serverscd.py:221, 263` là `1e-6`),
     trong khi `L1455` nêu mức sàn độ lệch chuẩn toàn cục $10^{-4}$ bằng lời nhưng không gắn với ký hiệu $\epsilon_\sigma$ ở `L864`.
  2. **$\kappa$ không phải là hằng số $\epsilon$ vô cùng bé**: Trong phương trình `(\ref{eq:raw_conc})` (`Eq. (21)`), khi $n_t^c = 1$, thừa số mẫu là $\log(1 + \kappa)$. Nếu hiểu nhầm $\kappa \to 0^+$, $\log(1+\kappa) \sim \kappa \to 0^+$ sẽ làm nổ giá trị $r_t^c \propto 1/\kappa$; còn việc chọn $\kappa = 1$, $\kappa = e - 1$ hay $\kappa = 10$ (như trong code `np.log(max(total_count, 1) + 10.0)`) làm thay đổi đáng kể tỷ số tương đối $r_t^c / \bar{r}_t$ giữa lớp ít mẫu và lớp nhiều mẫu. Thiếu $\kappa = 10$ và cơ số logarit tự nhiên $\ln$, phương trình `(21)`–`(22)` không thể tái lập toán học.
  3. **Thiếu điều kiện miền xác định tại nơi giới thiệu tham số**: Bài báo bỏ sót ràng buộc $0 \le \gamma_{\max} \le 1$ và $n_0 > 0$ tại `(\ref{eq:anchor})` (`L844–L850`, cần thiết để $\gamma^c \in [0, 1)$ xác định tại $n_k^c = 0$ và $p^c$ là tổ hợp lồi); bỏ sót $\epsilon_\mu \ge 0$, $\epsilon_\sigma, \epsilon_s, \epsilon_g, \epsilon > 0$, $\alpha_v \ge 0, \alpha_q > 0$ tại `L829–L911` (lưu ý $\epsilon_s > 0$ là bắt buộc ngay cả khi $c \in \mathcal{A}$, vì $p^c = (1-\gamma^c)\mu^c + \gamma^c\mu_k^c$ vẫn có thể triệt tiêu về $\mathbf{0}$ dù $\|\mu^c\|_2 > \epsilon_\mu$); và bỏ sót $\epsilon_v > 0, \epsilon_r > 0$ tại `L1179, L1228–L1256` (khẳng định tại `L1255` *"The variance floor makes $r_t^c$ positive for observed classes"* chỉ đúng khi và chỉ khi $\epsilon_v > 0$, còn tránh $0/0$ khi $\mu_t^c = \mathbf{0}$ đòi hỏi $\epsilon_r > 0$).
- **Đề xuất cách sửa cụ thể**:
  - Bổ sung điều kiện miền ngay tại Section III-D/E: thêm `where $0\leq\gamma_{\max}\leq 1$ and $n_0>0$` tại `L850`; `with positive stabilization constants $\epsilon_\sigma,\epsilon_s,\epsilon_g,\epsilon_{\mathrm{norm}}>0$, threshold $\epsilon_\mu\geq 0$, and $\alpha_v\geq 0,\alpha_q>0$` tại `L864`; `with $\epsilon_v>0$` tại `L1180`; và `where $\tau>0$, $\kappa>0$, $\epsilon_r>0$, and $0\leq\rho<1$` tại `L1252–L1254`.
  - Tại Section IV (`L1454–L1455`), bổ sung đầy đủ các giá trị thực tế đã xác minh từ mã nguồn:
    ```latex
    We set the support-smoothing offset $\kappa=10$ (using the natural logarithm $\ln$), anchor norm threshold $\epsilon_\mu=10^{-8}$, client-side global standard-deviation floor $\epsilon_\sigma=10^{-4}$, server variance floor $\epsilon_v=10^{-8}$, and numerical stabilizers $\epsilon_s=\epsilon_r=10^{-6}$, $\epsilon_g=10^{-8}$, and $\epsilon_{\mathrm{norm}}=10^{-12}$.
    ```

---

### `MATH-02` — Mâu thuẫn Nội bộ Đối với Lớp Đơn mẫu ($n_k^c = 1$) và Lớp Rỗng ($n_k^c = 0$) Giữa Phương trình `(2)`, Giả thiết Gauss Không Suy biến, Bản tin Tải lên Đặc $C(2d+1)$ và Quy tắc Fallback ở Mục IV
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L608–L636` (Section III-A, Eqs. `(1)`–`(2)`), `L1337–L1339` (Section III-F), `L1373–L1388` (Section III-G, Proposition~\ref{prop:exact}), và `L1455` (Section IV); đối chiếu `system/flcore/clients/clientscd.py` (`L120–L129`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L608-L632:
  For each locally active class with $n_k^c\geq 1$,
  the empirical moments are defined as follows,
  with the communication-round index omitted for brevity:
  ...
  (\sigma_k^c)^{\odot 2}
  &=
  \frac{1}{n_k^c}
  \sum_{(x,y)\in\mathcal{D}_k^c}
  \big(f_k(x)-\mu_k^c\big)^{\odot 2}.
  \label{eq:var}
  ...
  These means and diagonal variances define local
  diagonal Gaussian approximations.

  % Overleaf/ieee2026.tex, L1455:
  Local statistics are computed over complete training minibatches (excluding any incomplete final minibatch, so uploaded counts reflect processed samples), using a standard-deviation fallback of $10^{-6}$ for singleton classes and a client-side floor of $10^{-4}$ on global standard deviations.
  ```
- **Phân tích chi tiết**:
  1. **Mâu thuẫn trực tiếp tại $n_k^c = 1$**: Phương trình `(\ref{eq:var})` (`Eq. (2)`) được phát biểu cho mọi $n_k^c \ge 1$. Khi $n_k^c = 1$ (trường hợp lớp chỉ có 1 mẫu cục bộ, rất phổ biến dưới phân phối $\mathrm{Dir}(0.1)$ trên CIFAR-100, Flowers102 hay Tiny-ImageNet), mẫu duy nhất trùng với $\mu_k^c$, nên `Eq. (2)` bắt buộc cho $(\sigma_k^c)^{\odot 2} = \mathbf{0}$ ($\sigma_k^c = \mathbf{0}$). Điều này mâu thuẫn trực tiếp với `L1455` và `clientscd.py:125`, nơi các lớp $n_k^c = 1$ tải lên $\sigma_k^c = 10^{-6}\mathbf{1} \neq \mathbf{0}$. Khi có ít nhất một client với $n_k^c = 1$, giá trị $\sigma_k^c$ tải lên không thỏa mãn `Eq. (2)` và tạo ra độ lệch $+\frac{|\{k\in\mathcal{S}_t^c:n_k^c=1\}|}{\bar{n}_t^c}10^{-12}\mathbf{1}$ trong `(\ref{eq:server_var})` so với phương sai thực nghiệm chính xác ở Proposition~\ref{prop:exact}.
  2. **Ma trận hiệp phương sai suy biến**: Khi $\sigma_{k,j}^c = 0$ (theo `Eq. (2)` với $n_k^c = 1$, hoặc trên các chiều `ReLU` chết bằng $0$ với mọi mẫu), $\mathrm{diag}((\sigma_k^c)^{\odot 2})$ bị suy biến ($\det = 0$), làm xấp xỉ Gauss cục bộ ở `L631–L632` trở thành phân phối suy biến nếu không nói rõ bước chính quy hóa tọa độ. Hơn nữa, client truyền đi $\sigma_k^c$ (`L634`), nhưng `Eq. (2)` mới chỉ định nghĩa $(\sigma_k^c)^{\odot 2}$ chứ chưa định nghĩa tường minh phép lấy căn bậc hai từng phần tử.
  3. **Thiếu quy ước đệm không cho lớp vắng mặt ($n_k^c = 0$) khi xét độ phức tạp bản tin đặc**: `Eqs. (1)–(2)` chỉ định nghĩa cho $n_k^c \ge 1$, nhưng `L1337` khẳng định mỗi bản tin tải lên đầy đủ chứa đúng $C(2d+1)$ vô hướng cho cả $C$ lớp (`clientscd.py:128` gán bộ đệm không $(n_k^c, \mu_k^c, \sigma_k^c) = (0, \mathbf{0}, \mathbf{0})$ khi $n_k^c = 0$). Quy ước zero-padding này chưa được phát biểu ở Mục III-A.
- **Đề xuất cách sửa cụ thể**:
  Tách bạch phương sai thực nghiệm thô $(s_k^c)^{\odot 2}$ (phục vụ chứng minh Proposition~\ref{prop:exact}) với độ lệch chuẩn tải lên $\sigma_k^c$ có xử lý biên cho $n_k^c = 1$ và $n_k^c = 0$ tại `L611–L636`:
  ```latex
  \begin{align}
  \mu_k^c &= \frac{1}{n_k^c}\sum_{(x,y)\in\mathcal{D}_k^c} f_k(x), \label{eq:mean}\\
  (s_k^c)^{\odot 2} &= \frac{1}{n_k^c}\sum_{(x,y)\in\mathcal{D}_k^c}\big(f_k(x)-\mu_k^c\big)^{\odot 2}. \label{eq:var}
  \end{align}
  For communication, client $k$ forms the uploaded standard deviation $\sigma_k^c = \big((s_k^c)^{\odot 2}\big)^{\odot 1/2}$ when $n_k^c \geq 2$, applies a positive fallback $\sigma_k^c = \epsilon_{\mathrm{single}}\mathbf{1}$ ($\epsilon_{\mathrm{single}}=10^{-6}$) when $n_k^c = 1$, and zero-pads locally absent classes ($n_k^c = 0$) as $(0, \mathbf{0}, \mathbf{0})$ in dense $C(2d+1)$ messages (only positive-count entries $n_k^c > 0$ enter server pooling). Setting $\epsilon_{\mathrm{single}}=0$ recovers $(\sigma_k^c)^{\odot 2}=(s_k^c)^{\odot 2}$ used in Proposition~\ref{prop:exact}.
  ```

---

### `MATH-03` — Lỗi Gộp Bội số Tập hợp Vector (Set vs. Indexed Multiset) và Đụng độ Ba Ký hiệu ($p, n, j$) trong Hàm Mất mát Tương phản Lai $\mathcal{L}_{\mathrm{SC}}$ (`Eq. (13)`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L951–L978` (Section III-D, Eq. `(13)` / `\ref{eq:sc}`).
- **Trích dẫn nguyên văn**:
  ```latex
  To incorporate sample-level relationships,
  each valid sample $i\in\mathcal{I}$ is also contrasted
  against
  $\{\hat{z}_j:j\in\mathcal{I},j\neq i\}
  \cup\{q^c:c\in\mathcal{A}\}$.
  Let $\mathcal{P}_i$ and $\mathcal{N}_i$ denote
  the same-label and different-label references,
  respectively.
  ...
  \begin{equation}
  \begin{aligned}
  \mathcal{L}_{\mathrm{SC}}
  ={}&-\frac{1}{B}\sum_{i\in\mathcal{I}}
  \frac{1}{|\mathcal{P}_i|}
  \sum_{p\in\mathcal{P}_i}\\
  &\log
  \frac{\exp(\hat{z}_i^\top p/\tau)}
  {\exp(\hat{z}_i^\top p/\tau)
  +\sum_{n\in\mathcal{N}_i}
  \exp(\hat{z}_i^\top n/\tau)}.
  \end{aligned}
  \label{eq:sc}
  \end{equation}
  ```
- **Phân tích chi tiết**:
  1. **Lỗi định nghĩa tập hợp vector trong $\mathbb{R}^d$ thay vì họ có đánh chỉ số**: Biểu thức $\{\hat{z}_j : j \in \mathcal{I}, j \neq i\} \cup \{q^c : c \in \mathcal{A}\} \subset \mathbb{R}^d$ là một **tập hợp các vector thuần túy** trong $\mathbb{R}^d$, không phải một họ được đánh chỉ số trên tổng rời $(\mathcal{I}\setminus\{i\}) \sqcup \mathcal{A}$. Hệ quả toán học kép: (a) bản thân một vector thuộc $\mathbb{R}^d$ không mang nhãn lớp đi kèm, nên việc tách tập hợp đó thành tập "cùng nhãn" ($\mathcal{P}_i$) và "khác nhãn" ($\mathcal{N}_i$) không xác định chặt chẽ nếu hai phần tử khác nhãn tình cờ trùng vector chiếu; (b) bất cứ khi nào hai mẫu phân biệt $j_1 \neq j_2$ sinh ra cùng vector chiếu $\hat{z}_{j_1} = \hat{z}_{j_2}$ (ví dụ hai ảnh trùng lặp hoặc hai vector rơi vào vùng bão hòa) hoặc $\hat{z}_j = q^{y_i}$, ký hiệu tập hợp toán học $\{ \cdot \}$ tự động gộp các phần tử trùng nhau về bội số $1$, làm thay đổi $|\mathcal{P}_i|$ và tổng mẫu số so với phép nhân ma trận độ tương đồng kích thước $|\mathcal{I}| \times (|\mathcal{I}| + |\mathcal{A}|)$ trong mã nguồn (`clientscd.py:310–345` và `L1347–L1350`).
  2. **Đụng độ 3 ký hiệu ngay trong cùng một mục**:
     - Biến chạy vector dương $p \in \mathcal{P}_i$ (trong không gian đã chiếu chuẩn hóa) đụng độ trực tiếp với mỏ neo thích ứng client $p^c \in \mathbb{R}^d$ (không gian chưa chiếu, `L844`) và hàm mật độ xác suất $p_c(z)$ (`L689`).
     - Biến chạy vector âm $n \in \mathcal{N}_i$ đụng độ với toàn bộ họ ký hiệu số đếm mẫu nguyên $n_k, n_k^c, \bar{n}_t^c, n_t^c, n_0$.
     - Chỉ số mẫu $j$ trong $\hat{z}_j \in \mathbb{R}^d$ (`L954`) đụng độ với chỉ số tọa độ $j \in \{1,\dots,d\}$ trong $z_j \in \mathbb{R}$ chỉ 35 dòng bên dưới (`L989`).
- **Đề xuất cách sửa cụ thể**:
  Đánh chỉ số tường minh trên tập chỉ số hợp rời $\mathcal{R}_i = (\mathcal{I}\setminus\{i\}) \sqcup \mathcal{A}$ và dùng chỉ số mẫu $\ell$, biến chạy chỉ số $r, s$:
  ```latex
  To incorporate sample-level relationships, define the indexed reference family $\{(v_r, \tilde{y}_r)\}_{r\in\mathcal{R}_i}$ over the disjoint index union $\mathcal{R}_i = (\mathcal{I}\setminus\{i\})\sqcup\mathcal{A}$, where $(v_\ell,\tilde{y}_\ell)=(\hat{z}_\ell, y_\ell)$ for sample indices $\ell\in\mathcal{I}\setminus\{i\}$ and $(v_c,\tilde{y}_c)=(q^c, c)$ for anchor indices $c\in\mathcal{A}$. Let $\mathcal{P}_i=\{r\in\mathcal{R}_i:\tilde{y}_r=y_i\}$ and $\mathcal{N}_i=\{r\in\mathcal{R}_i:\tilde{y}_r\neq y_i\}$ denote the positive and negative reference index sets (note $|\mathcal{P}_i|\geq 1$ for every $i\in\mathcal{I}$ since $y_i\in\mathcal{A}$, and $|\mathcal{N}_i|\geq 1$ whenever $|\mathcal{A}|\geq 2$). Using base temperature $\tau>0$, the hybrid instance--prototype contrastive loss is
  \begin{equation}
  \begin{aligned}
  \mathcal{L}_{\mathrm{SC}}
  ={}&-\frac{1}{B}\sum_{i\in\mathcal{I}}
  \frac{1}{|\mathcal{P}_i|}
  \sum_{r\in\mathcal{P}_i}\\
  &\ln
  \frac{\exp(\hat{z}_i^\top v_r/\tau)}
  {\exp(\hat{z}_i^\top v_r/\tau)
  +\sum_{s\in\mathcal{N}_i}
  \exp(\hat{z}_i^\top v_s/\tau)}.
  \end{aligned}
  \label{eq:sc}
  \end{equation}
  ```

---

### `MATH-04` — Chưa Định nghĩa Đặc trưng Ảo Có Chỉ số Mẫu $\widetilde{z}_i^{y_i,\mathrm{loc}}$, Đụng độ Ký hiệu Nhiễu $\epsilon$, Mẫu số Không Được Bảo vệ trong `Eq. (17)` và Biểu thức Hệ số Ramp-Up $\lambda_{k,t}$ Chưa Đóng Kín
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L797` (`Fig. 2` caption), `L910–L911`, `L1039–L1050` & `L1074–L1110` (Section III-D, Eqs. `(16)`–`(17)`). *(Xem thêm `ALGO-03` ở Phần B về lỗi gộp cổng $\lambda > 0$ trong `Algorithm 1` và `clientscd.py`.)*
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1039-L1048:
  Here, the ProtoNCE and compactness terms have
  unit relative weights, as in the default configuration.
  ...
  After warm-up, \textcolor{red}{when $|\mathcal{A}|\geq2$,
  we use
  $\lambda_{k,t}
  =\lambda\min\{1,(t_k-T_{\mathrm{warm}})
   /T_{\mathrm{ramp}}\}$;
  otherwise its effective value is zero.}

  % Overleaf/ieee2026.tex, L1077-L1109:
  \[
  \widetilde{z}^{c,b}
  =
  \Pi\big(
  p^c+\eta_b\epsilon\odot\widetilde{\sigma}^c
  \big),
  \qquad
  \epsilon\sim\mathcal{N}(0,\mathbf{I}),
  \]
  ...
  \begin{equation}
  \begin{aligned}
  \mathcal{L}_{\mathrm{head}}
  ={}&\mathcal{L}_{\mathrm{data}}\\
  &+\frac{\lambda_{\mathrm{vl}}}{|\mathcal{I}|}
  \sum_{i\in\mathcal{I}}
  \ell_{\mathrm{CE}}
  \big(h_k(\widetilde{z}_i^{y_i,\mathrm{loc}}),y_i\big)\\
  &+\frac{\lambda_{\mathrm{vg}}}{|\mathcal{A}|}
  \sum_{c\in\mathcal{A}}
  \ell_{\mathrm{CE}}
  \big(h_k(\widetilde{z}^{c,\mathrm{glob}}),c\big).
  \end{aligned}
  \label{eq:head_loss}
  \end{equation}
  ... Both virtual terms are disabled during warm-up or when usable global anchors are insufficient.
  ```
- **Phân tích chi tiết**:
  1. **Chỉ số dưới $i$ chưa được định nghĩa trong $\widetilde{z}_i^{y_i,\mathrm{loc}}$**: Dòng `L1078` định nghĩa công thức chung $\widetilde{z}^{c,b}$ không hề có chỉ số mẫu $i$ và chỉ viết một biến ngẫu nhiên đơn lẻ $\epsilon \sim \mathcal{N}(0, \mathbf{I})$, nhưng ngay tại `Eq. (17)` (`L1097`) lại gọi $\widetilde{z}_i^{y_i,\mathrm{loc}}$ với chỉ số $i$ chưa từng được giải thích. Cần chỉ rõ các vector nhiễu Gauss chuẩn được lấy mẫu độc lập từng mẫu $i \in \mathcal{I}$ ở nhánh cục bộ và từng lớp $c \in \mathcal{A}$ ở nhánh toàn cục.
  2. **Đụng độ ký hiệu $\epsilon$ ngay trong cùng Mục III-D**: `L797` và `L1084` dùng $\epsilon \in \mathbb{R}^d$ làm vector ngẫu nhiên Gauss nhiều chiều, trùng tên hoàn toàn với hằng số vô hướng tất định $\epsilon > 0$ trong phép chuẩn hóa $\operatorname{norm}_{\epsilon}(a) = a/\max(\|a\|_2, \epsilon)$ tại `L910–L911`.
  3. **Chia cho $0$ và thiếu hàm chỉ báo đóng/mở trong `Eq. (17)`**: Khác với `Eq. (16)` có hệ số $\lambda_{k,t}$ điều khiển trực tiếp, `Eq. (17)` viết trần $\frac{\lambda_{\mathrm{vl}}}{|\mathcal{I}|}$ và $\frac{\lambda_{\mathrm{vg}}}{|\mathcal{A}|}$ không có hàm chỉ báo (indicator), gây phép chia vô định $0/0$ khi $\mathcal{I} = \emptyset$ hoặc $\mathcal{A} = \emptyset$, mà chỉ dựa vào lời văn giải thích phía sau (`L1105–L1109`).
  4. **Công thức $\lambda_{k,t}$ chưa tự đóng kín và câu chữ điều kiện thiếu chặt chẽ (`L1039–L1048`, `L1108–L1109`)**: Biểu thức $\lambda\min\{1, (t_k - T_{\mathrm{warm}})/T_{\mathrm{ramp}}\}$ không có $\max\{0, \cdot\}$ hay $\operatorname{clip}(\cdot, 0, 1)$, nên nếu đọc độc lập sẽ nhận giá trị âm khi $t_k < T_{\mathrm{warm}}$. Ngoài ra, cụm `"as in the default configuration"` (`L1040`) làm yếu định nghĩa phương trình `(16)`, đại từ `"its"` (`L1048`) không có tiền ngữ danh từ rõ ràng, `"After warm-up"` diễn đạt bằng lời thay vì $t_k > T_{\mathrm{warm}}$, và cụm `"when usable global anchors are insufficient"` (`L1109`) không ghi rõ ngưỡng toán học $|\mathcal{A}| < 2$.
- **Đề xuất cách sửa cụ thể**:
  Đổi hằng số chuẩn hóa ở `L910` thành $\epsilon_{\mathrm{norm}}$, ký hiệu nhiễu Gauss là $\zeta_i^{\mathrm{loc}}, \zeta^{c,\mathrm{glob}} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, đóng kín công thức $\lambda_{k,t}$ và đưa hàm chỉ báo $\mathbb{I}_{k,t}^{\mathrm{stat}} = \mathbb{I}(t_k > T_{\mathrm{warm}} \wedge |\mathcal{A}| \ge 2)$ vào trực tiếp `Eq. (17)`:
  ```latex
  % Tại L1039-L1048:
  Here, $\mathcal{L}_{\mathrm{NCE}}$ and $\mathcal{L}_{\mathrm{PL}}$ carry unit relative weights, while $\lambda_{\mathrm{ins}}\geq 0$ scales the hybrid contrastive term. Let $t_k$ denote client $k$'s cumulative participation count, including the current round. With $T_{\mathrm{warm}}\geq 0$ and $T_{\mathrm{ramp}}>0$, define $\mathbb{I}_{k,t}^{\mathrm{stat}}=\mathbb{I}(t_k>T_{\mathrm{warm}}\wedge |\mathcal{A}|\geq 2)$ and set
  \[
  \lambda_{k,t} = \lambda\,\mathbb{I}_{k,t}^{\mathrm{stat}}\,\operatorname{clip}\!\left(\frac{t_k-T_{\mathrm{warm}}}{T_{\mathrm{ramp}}},\,0,\,1\right).
  \]

  % Tại L1077-L1104:
  \begin{equation}
  \widetilde{z}_i^{y_i,\mathrm{loc}} = \Pi\big(p^{y_i}+\eta_{\mathrm{loc}}\zeta_i^{\mathrm{loc}}\odot\widetilde{\sigma}^{y_i}\big), \qquad
  \widetilde{z}^{c,\mathrm{glob}} = \Pi\big(p^c+\eta_{\mathrm{glob}}\zeta^{c,\mathrm{glob}}\odot\widetilde{\sigma}^c\big),
  \label{eq:virt_feat}
  \end{equation}
  with independent standard Gaussian vectors $\zeta_i^{\mathrm{loc}}, \zeta^{c,\mathrm{glob}} \sim \mathcal{N}(\mathbf{0},\mathbf{I}_d)$. The head objective is
  \begin{equation}
  \begin{aligned}
  \mathcal{L}_{\mathrm{head}}
  ={}&\mathcal{L}_{\mathrm{data}}
  +\mathbb{I}_{k,t}^{\mathrm{stat}}\mathbb{I}(|\mathcal{I}|>0)\frac{\lambda_{\mathrm{vl}}}{|\mathcal{I}|}
  \sum_{i\in\mathcal{I}}
  \ell_{\mathrm{CE}}\big(h_k(\widetilde{z}_i^{y_i,\mathrm{loc}}),y_i\big)\\
  &+\mathbb{I}_{k,t}^{\mathrm{stat}}\frac{\lambda_{\mathrm{vg}}}{|\mathcal{A}|}
  \sum_{c\in\mathcal{A}}
  \ell_{\mathrm{CE}}\big(h_k(\widetilde{z}^{c,\mathrm{glob}}),c\big).
  \end{aligned}
  \label{eq:head_loss}
  \end{equation}
  ```

---

### `MATH-05` — Chưa Đặc tả Tường minh Công thức Chẩn đoán Hình học $D_B^{\mathrm{eval}}(c,c')$ (và Cách Chọn Tâm $a^c$) ở Mục IV, Cùng Biến thể `"Class-Specific Projection"` Chưa Định nghĩa ở Bảng III
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1464` (Section IV, *Metrics*) và `L1608`, `L1620` (Section V-C, `Table III`); đối chiếu trực tiếp với `system/utils/motivation_metrics.py` (`L92, L125, L138–L148, L161–L185`) và `system/flcore/clients/clientscd.py` (`L222, L276–L284`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1464:
  For representation geometry, we evaluate the mean nearest-other-anchor Euclidean margin and a dimension-rescaled Gaussian overlap score averaged over all distinct class pairs, where each pair contributes $\exp(-D_B^{\mathrm{eval}}(c,c')/\sqrt{d})$. Here, $D_B^{\mathrm{eval}}(c,c')$ computes the Bhattacharyya distance from count-pooled class statistics prior to EMA smoothing, with coordinate-wise pooled variances floored at $10^{-4}$ and stabilized by $10^{-6}$; this rescaled diagnostic is distinct from the unscaled Bhattacharyya coefficient in (\ref{eq:bc}).

  % Overleaf/ieee2026.tex, L1608 (Table III):
  \quad Class-specific projection (vs. Shared)            & 75.08 & -1.58 \\
  ```
- **Phân tích chi tiết**:
  1. **Mô tả bằng lời cho $D_B^{\mathrm{eval}}(c,c')$ và $a^c$ không xác định duy nhất về mặt đại số**: Phương trình Bhattacharyya `(\ref{eq:bhatt})` có hai tổng phụ thuộc phương sai khác nhau (phân thức khoảng cách trung bình chia cho $\bar{\sigma}_j^2$ và số hạng logarit tỷ số phương sai). Câu văn *"coordinate-wise pooled variances floored at $10^{-4}$ and stabilized by $10^{-6}$"* không cho phép người đọc biết $10^{-6}$ được cộng vào từng phương sai lớp trước khi lấy trung bình hay cộng riêng vào mẫu số phân thức/logarit (các cách đặt này cho giá trị số khác nhau trên $d=512$ chiều). Kiểm tra trực tiếp mã nguồn `system/utils/motivation_metrics.py` (`L92, L125, L161–L185`) xác nhận công thức chính xác trong code là: trước hết tính $\tilde{v}_{t,j}^c = \max(\bar{v}_{t,j}^c, 10^{-4}) + 10^{-6}$, đặt $\tilde{\sigma}_{t,j}^{c,c'\,2} = \tfrac{1}{2}(\tilde{v}_{t,j}^c + \tilde{v}_{t,j}^{c'})$, rồi tính $D_B^{\mathrm{eval}}(c,c')$ mà **không cộng thêm $10^{-6}$ lần nữa** bên trong phân thức hay hàm $\ln(\cdot)$ (đồng thời vì $\tilde{v}_{t,j}^c \ge 1.01\times 10^{-4} > 10^{-4}$ với mọi tọa độ, nên `d_eff = d = 512`). Ngoài ra, `L1464` cũng chưa nêu rõ vector mỏ neo $a^c$ dùng để tính biên Euclid $\min_{c'\neq c}\|a^c - a^{c'}\|_2$ là gì: trong `motivation_metrics.py:138–148`, `FedSCD` dùng trung bình đã làm trơn EMA $a^c = \mu_t^c$, `FedTGP`/`FedSA`/`FedORGP` dùng `global_protos`, còn `FedCOF`/`FedGMKD` dùng trung bình gộp trước EMA $\bar{\mu}_t^c$.
  2. **Thiếu phương trình toán học cho biến thể `"Class-specific projection"` (`L1608, L1620`)**: Khi bỏ vector trọng số chung $w$ trong `Eq. (11)` (`\ref{eq:proj}`), bài báo không hề định nghĩa cách tính tích vô hướng khi so sánh mẫu $z_i$ (nhãn $y_i$) với mỏ neo lớp ứng viên $c \neq y_i$ trong $\mathcal{L}_{\mathrm{NCE}}$ hoặc với mẫu khác $z_\ell$ ($y_\ell \neq y_i$) trong $\mathcal{L}_{\mathrm{SC}}$. Đối chiếu `clientscd.py:222, 276–284` (`--proj_mode class`) cho thấy biến thể này chuẩn hóa riêng từng lớp $w^c = u^c / (\operatorname{mean}(u^c) + \epsilon_g)$, sau đó chiếu mỗi mẫu $z_i$ bằng trọng số của nhãn thật $w^{y_i}$ ($\hat{z}_i = \operatorname{norm}_{\epsilon_{\mathrm{norm}}}((z_i - m)\odot w^{y_i})$) và chiếu mỗi mỏ neo $p^c$ bằng $w^c$ ($q^c = \operatorname{norm}_{\epsilon_{\mathrm{norm}}}((p^c - m)\odot w^c)$), dẫn tới tích vô hướng $\hat{z}_i^\top q^c$ bị tính giữa hai hệ metric tọa độ khác nhau khi $c \neq y_i$.
- **Đề xuất cách sửa cụ thể**:
  - Bổ sung công thức chính xác khớp 100% với `motivation_metrics.py` vào Section IV (`L1464`):
    ```latex
    Specifically, for currently observed classes $\mathcal{C}_t^{\mathrm{cur}}=\{c:\bar{n}_t^c>0\}$, nearest-class Euclidean margins $\frac{1}{|\mathcal{C}_t^{\mathrm{cur}}|}\sum_{c\in\mathcal{C}_t^{\mathrm{cur}}}\min_{c'\in\mathcal{C}_t^{\mathrm{cur}}\setminus\{c\}}\|a^c-a^{c'}\|_2$ are evaluated using each method's global class anchors ($a^c=\mu_t^c$ for FedSCD, learned/aggregated global prototypes for FedTGP, FedSA, and FedORGP, and count-pooled means $\bar{\mu}_t^c$ for FedCOF and FedGMKD). For distributional overlap, letting $\tilde{v}_{t,j}^c = \max(\bar{v}_{t,j}^c, 10^{-4}) + 10^{-6}$ and $\tilde{\sigma}_{t,j}^{c,c'\,2} = \tfrac{1}{2}(\tilde{v}_{t,j}^c + \tilde{v}_{t,j}^{c'})$, we compute
    \[
    D_B^{\mathrm{eval}}(c,c') = \frac{1}{8}\sum_{j=1}^{d}\frac{(\bar{\mu}_{t,j}^c - \bar{\mu}_{t,j}^{c'})^2}{\tilde{\sigma}_{t,j}^{c,c'\,2}} + \frac{1}{2}\sum_{j=1}^{d}\left(\ln\tilde{\sigma}_{t,j}^{c,c'\,2} - \frac{1}{2}\ln\tilde{v}_{t,j}^c - \frac{1}{2}\ln\tilde{v}_{t,j}^{c'}\right).
    \]
    ```
  - Bổ sung định nghĩa toán học ngắn gọn tại Section V-C (`L1620`):
    ```latex
    Specifically, the class-specific projection variant replaces the shared weight $w$ in (\ref{eq:proj}) with $w^{y_i}=u^{y_i}/(\operatorname{mean}(u^{y_i})+\epsilon_g)$ when projecting sample $z_i$ and $w^c=u^c/(\operatorname{mean}(u^c)+\epsilon_g)$ when projecting anchor $p^c$.
    ```

---

### `MATH-06` — Đụng độ Ký hiệu Nội Phương trình & Liên Tiểu mục ($d^c(z)$ vs. Số chiều $d$, $z_j$ Mẫu vs. Tọa độ, $m$ vs. $m_j^c$, $\bar{\sigma}_j^2$ vs. $\bar{\sigma}_t^c$, $B_t^c$ vs. $B$)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L670–L683` (`Eq. (3)`), `L908` (`Eq. (11)`), `L934` & `L954`, `L986–L1006` (`Eq. (14)`), `L1176`, và `L1396` (Proposition~\ref{prop:drift}).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L986-L990 (Eq. 14):
  d^c(z)
  &=
  \frac{1}{d}\sum_{j=1}^{d}
  \frac{g_j^c(z_j-p_j^c)^2}{v_j^c},
  \label{eq:dist}
  ```
- **Phân tích chi tiết**:
  1. **Đụng độ ký hiệu $d$ ngay trong cùng phương trình `(14)` (`L986–L989`)**: Ký tự $d$ vừa biểu diễn hàm phạt độ nén $d^c : \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ ở vế trái, vừa biểu diễn số chiều không gian đặc trưng $d \in \mathbb{N}$ trong $\frac{1}{d}\sum_{j=1}^d$ ở vế phải (hơn nữa $d^c(z)$ là bình phương khoảng cách có trọng số nên không thỏa mãn bất đẳng thức tam giác của một metric $d(\cdot,\cdot)$).
  2. **Xung đột ngữ nghĩa chỉ số dưới của $z$ (`L934, L954` so với `L989, L1000`)**: Ở `L934` ($z_i \in \mathbb{R}^d$), `L954` ($\hat{z}_j \in \mathbb{R}^d$) và `L994` ($z_i \in \mathbb{R}^d$), chỉ số dưới của $z$ là **chỉ số mẫu trong minibatch**; nhưng ngay tại `L989` và `L999–L1005`, cùng cách viết $z_j \in \mathbb{R}$ lại biểu diễn **tọa độ vô hướng thứ $j$** của một vector $z \in \mathbb{R}^d$.
  3. **Đụng độ ký hiệu $m$ ngay trong Section III-D (`L908` vs. `L999`)**: Tại `L908`, $m = |\mathcal{A}|^{-1}\sum_{c\in\mathcal{A}} p^c \in \mathbb{R}^d$ là tâm trung bình của các mỏ neo lớp, nhưng tại `L999` trong cùng tiểu mục, $m_j^c = \mathbb{E}[z_j \mid y=c]$ lại là kỳ vọng quần thể có điều kiện theo lớp của client.
  4. **Đụng độ ký hiệu $\bar{\sigma}$ và $B$**: $\bar{\sigma}_j^2 = \tfrac{1}{2}((\sigma_j^c)^2 + (\sigma_j^{c'})^2)$ trong `Eq. (3)` (`L681`) ẩn sự phụ thuộc vào cặp lớp $(c,c')$ và trùng ký hiệu gạch ngang với độ lệch chuẩn gộp phía server $\bar{\sigma}_t^c$ (`L1176`); vector độ phân tán liên client $B_t^c \in \mathbb{R}^d$ tại `L1396` dùng lại chữ $B$ vốn là kích thước minibatch xuyên suốt Section III-D/F.
- **Đề xuất cách sửa cụ thể**:
  - Đổi $d^c(z)$ trong `Eqs. (14)–(15)` thành $\psi^c(z)$ (hoặc $M^c(z)$), viết trích xuất tọa độ dưới dạng $[z]_j$ (hoặc dùng $\ell$ cho mẫu và $j$ riêng cho tọa độ):
    ```latex
    \psi^c(z) = \frac{1}{d}\sum_{j=1}^{d}\frac{g_j^c([z]_j - p_j^c)^2}{v_j^c}, \qquad \mathcal{L}_{\mathrm{PL}} = \frac{1}{B}\sum_{i\in\mathcal{I}}\psi^{y_i}(z_i).
    ```
  - Đổi kỳ vọng có điều kiện $m_j^c$ (`L999`) thành $\nu_{k,j}^c = \mathbb{E}_k[[z]_j \mid y=c]$, viết $\bar{\sigma}_{c,c',j}^2 = \tfrac{1}{2}((\sigma_j^c)^2 + (\sigma_j^{c'})^2)$ trong `Eq. (3)`, và đổi $(W_t^c, B_t^c)$ trong Proposition~\ref{prop:drift} thành $(S_{t,\mathrm{within}}^c, S_{t,\mathrm{between}}^c)$ nếu muốn tránh trùng chữ $B$.

---

### `MATH-07` — Nhập nhằng Ký hiệu Giữa Thống kê Cục bộ Lưu Cache Từ Vòng Trước và Thống kê Cục bộ Tính Mới Cuối Vòng $(\mu_k^c, n_k^c)$ Cùng Việc Ẩn Chỉ số Client $k$ Chưa Khai báo
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L611–L623` (`Eqs. (1)–(2)`), `L823–L825` & `L843–L857` (`Eq. (5)` / `\ref{eq:anchor}`), `L1144–L1165` (`Eqs. (18)–(19)`), và `L1454`.
- **Trích dẫn nguyên văn**:
  ```latex
  \begin{equation}
  p^c=(1-\gamma^c)\mu^c+\gamma^c\mu_k^c,
  \qquad
  \gamma^c=\gamma_{\max}
  \frac{n_k^c}{n_k^c+n_0},
  \label{eq:anchor}
  \end{equation}
  where $(\mu_k^c,n_k^c)$ are the client's most recently
  stored local statistics.
  ```
- **Phân tích chi tiết**:
  1. **Cùng một ký hiệu $(\mu_k^c, n_k^c)$ mang hai giá trị khác nhau trong cùng vòng $t$**: Tại đầu vòng $t$ (*trước* khi cập nhật SGD cục bộ, `Algorithm 1` `L1301`), phương trình `(\ref{eq:anchor})` (`L844`) sử dụng thống kê cục bộ **được lưu cache từ lần tham gia trước của client $k$** ($\tau_k < t$, tính trên tham số bộ trích xuất cũ); nhưng ở cuối vòng $t$ (*sau* khi cập nhật SGD, `Algorithm 1` `L1316–L1321`), các phương trình `(1)–(2)` và `(18)–(19)` lại dùng **chính ký hiệu $(\mu_k^c, n_k^c)$ không chỉ số thời gian đó** để chỉ thống kê vừa tính mới của vòng $t$.
  2. **Ẩn chỉ số client $k$ mà không thông báo**: Dòng `L823–L825` chỉ nói rằng chỉ số vòng $t$ được lược bỏ trên $(\mu^c, \sigma^c, \phi^c)$, nhưng không nói rằng chỉ số client $k$ cũng đang bị lược bỏ trên hàng loạt đại lượng phụ thuộc từng client: $\gamma^c, p^c, n_0, s^c, \widetilde{g}^c, g^c, u^c, \bar{u}, w, m, q^c$ (lưu ý nếu viết $n_0 = \max(5, |\mathcal{D}_k|/(2C))$ như ở `L1454` thì $n_0$ phụ thuộc trực tiếp vào kích thước tập dữ liệu $|\mathcal{D}_k|$ của client $k$), đồng thời trước `(\ref{eq:anchor})` chưa ghi rõ miền `for each $c \in \mathcal{A}$`.
- **Đề xuất cách sửa cụ thể**:
  Ký hiệu thống kê lưu cache từ vòng trước trong `(\ref{eq:anchor})` là $(\breve{\mu}_k^c, \breve{n}_k^c)$ (hoặc $(\mu_{k,\mathrm{prev}}^c, n_{k,\mathrm{prev}}^c)$), bổ sung câu khai báo tại `L825` rằng chỉ số client $k$ được lược bỏ trên $p^c, \gamma^c, s^c, \widetilde{g}^c, g^c, u^c, \bar{u}, w, m, q^c$ trong phạm vi Section III-D, và thêm `for each $c\in\mathcal{A}$,` ngay trước `(\ref{eq:anchor})`.

---

### `MATH-08` — Thiếu Nhất quán Toán tử Từng Phần tử, Cộng Vô hướng–Vector và Ký hiệu Phân phối Gauss Chuẩn Đa chiều (`0.5 + 0.5s^c` Thiếu $\mathbf{1}$, Dấu Phân số Quá tải vs. $\oslash$, Thiếu Khai báo `max` Từng Phần tử, và $\mathcal{N}(0,\mathbf{I})$ Viết Số `0` Vô hướng)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L797` (`Fig. 2` caption), `L833–L834`, `L863–L906`, `L1084` (Section III-D), và `L1176–L1180` (Section III-E).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L880-L890:
  \begin{align}
  s^c &=
  \frac{(p^c)^{\odot 2}}
  {\operatorname{mean}((p^c)^{\odot 2})+\epsilon_s},\\
  \widetilde{g}^c &=
  \frac{0.5+0.5s^c}{1+0.5s^c},
  \quad
  g^c =
  \frac{\widetilde{g}^c}
  {\operatorname{mean}(\widetilde{g}^c)+\epsilon_g}.
  \end{align}

  % Overleaf/ieee2026.tex, L797 & L1084:
  \epsilon\sim\mathcal{N}(0,\mathbf I)   % L797
  \epsilon\sim\mathcal{N}(0,\mathbf{I})  % L1084
  ```
- **Phân tích chi tiết**:
  1. **Không nhất quán khi cộng vô hướng với vector trong $\mathbb{R}^d$**: Tại `L864` (`\max(\sigma^c,\epsilon_\sigma\mathbf{1})`) và `L867` (`v^c=(\widetilde{\sigma}^c)^{\odot 2}+\tau_v\mathbf{1}`), bài báo nhân tường minh hằng số vô hướng với vector toàn số một $\mathbf{1}\in\mathbb{R}^d$; nhưng ngay dòng `L885` lại viết `0.5 + 0.5s^c` và `1 + 0.5s^c`, cộng thẳng số thực vô hướng với vector $s^c \in \mathbb{R}^d$ mà không nhân $\mathbf{1}$.
  2. **Quá tải thanh phân số `\frac{\cdot}{\cdot}`**: Trong `L880–L906`, cùng ký hiệu phân số `\frac` vừa dùng để chia vector cho một số thực vô hướng (`s^c`, `g^c`, `\bar{u}`, `w`), vừa dùng để chia từng phần tử Hadamard giữa hai vector thuộc $\mathbb{R}^d$ (`\widetilde{g}^c` tại `L885`, đáng lẽ nên dùng $\oslash$ để phân biệt rõ).
  3. **Chưa khai báo $\max(\cdot,\cdot)$ từng phần tử và chưa viết công thức $\operatorname{mean}(\cdot)$**: `L833–L834` chỉ nói *"Vector powers and divisions are interpreted element-wise"*, bỏ quên phép lấy cực đại từng tọa độ $\max(\cdot,\cdot)$ ở `L864, L1179, L1324`; còn $\operatorname{mean}(\cdot)$ ở `L878` chỉ mô tả bằng lời thay vì $\operatorname{mean}(a) = \frac{1}{d}\sum_{j=1}^d [a]_j$.
  4. **Ký hiệu trung bình không và ma trận đơn vị trong phân phối Gauss chuẩn (`L797, L1084`)**: Toàn bài dùng chữ in đậm cho các vector/ma trận hằng số ($\mathbf{1}, \mathbf{I}$), nhưng ở `L797` và `L1084` lại viết kỳ vọng $d$ chiều trong $\mathcal{N}(0, \mathbf{I})$ là số `0` không in đậm, thiếu chỉ số chiều $d$ trên $\mathbf{I}_d$, và tại `L797` viết `\mathbf I` thiếu cặp ngoặc nhọn `{I}`.
- **Đề xuất cách sửa cụ thể**:
  Cập nhật `L833–L834` thành *"Vector powers, Hadamard divisions ($\oslash$), and $\max(\cdot,\cdot)$ are applied coordinate-wise"*, định nghĩa $\operatorname{mean}(a) = \frac{1}{d}\sum_{j=1}^d [a]_j$, viết lại `L885` là
  ```latex
  \widetilde{g}^c = (0.5\mathbf{1} + 0.5s^c) \oslash (\mathbf{1} + 0.5s^c),
  ```
  và chuẩn hóa `L797` cùng `L1084` thành `\mathcal{N}(\mathbf{0}, \mathbf{I}_d)`.

---

### `MATH-09` — Thiếu Đồng nhất Ký hiệu Logarit (`\ln` vs. `\log`), Thiếu Điều kiện Dương cho Tiên nghiệm ở `L697`, Lược bỏ Miền Tích phân/Tổng & Công thức $\mathcal{L}_{\mathrm{mix}}$, và Chưa Định nghĩa Toán tử (`\operatorname{sg}`, `\operatorname{clip}`, `\operatorname{round}`)
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L676` vs. `L942, L969, L1229`; `L689` & `L696–L701` vs. `L1412` & `L1427–L1430`; `L1054–L1065`; `L1200`; `L1247`; `L1379–L1387`. *(Lưu ý: lỗi đánh số `Proposition III.1`–`III.3` tại `L43–L51` được gộp và phân tích chi tiết tại `IEEE-04` trong Phần C.)*
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L696-L701:
  \textcolor{red}{For the binary problem restricted to these classes,
  with normalized priors $\pi_c+\pi_{c'}=1$,
  the Bayes error satisfies
  $P_e(c,c')\leq
  \sqrt{\pi_c\pi_{c'}}\,\mathrm{BC}(c,c')$
  (Proposition~\ref{prop:bayes_bound}).}

  % Overleaf/ieee2026.tex, L1054-L1064:
  Writing $r_i=\operatorname{sg}[z_i]$, define $\mathcal{L}_{\mathrm{real}}=B^{-1}\sum_i\ell_{\mathrm{CE}}(h_k(r_i),y_i)$. ... $\mathcal{L}_{\mathrm{mix}}$ averages the cross-entropy losses for labels $y_i$ and $y_{\pi(i)}$ with weights $\xi$ and $1-\xi$.
  ```
- **Phân tích chi tiết**:
  1. **Trộn lẫn `\ln` và `\log`**: Phương trình `(\ref{eq:bhatt})` (`L676`) dùng `\ln`, trong khi `(\ref{eq:proto_nce})` (`L942`), `(\ref{eq:sc})` (`L969`) và `(\ref{eq:raw_conc})` (`L1229`) lại dùng `\log` cho cùng logarit tự nhiên cơ số $e$ mà không chú thích cơ số.
  2. **Thiếu điều kiện dương $\pi_c, \pi_{c'} > 0$ tại `L697`**: Chỉ viết $\pi_c + \pi_{c'} = 1$ mà thiếu $\pi_c, \pi_{c'} > 0$ (hoặc $\ge 0$) thì căn thức $\sqrt{\pi_c\pi_{c'}}$ chưa được bảo đảm thực (trong khi ở Proposition~\ref{prop:bayes_bound} tại `L1412` đã viết đúng là `positive priors`).
  3. **Lược bỏ miền lấy tích phân/tổng và viết $\mathcal{L}_{\mathrm{mix}}$ bằng lời**: Tích phân $\int \dots dz$ ở `L689` và `L1428–L1429` thiếu miền $\mathbb{R}^d$; tổng $\sum_i$ ở `L1056` và $\sum_k\sum_i$ trong chứng minh Proposition~\ref{prop:exact} (`L1379–L1385`) thiếu cận dưới/trên; hàm $\mathcal{L}_{\mathrm{mix}}$ (`L1062–L1064`) chỉ mô tả bằng văn xuôi thay vì biểu thức toán học tường minh.
  4. **Toán tử chưa định nghĩa khi xuất hiện lần đầu**: $\operatorname{sg}[\cdot]$ (stop-gradient, `L1054`), $\operatorname{clip}(x,a,b) = \min(\max(x,a),b)$ (`L1247`), và quy tắc làm tròn điểm giữa của $\operatorname{round}(\cdot)$ (`L1200` — trong Python/NumPy là round-to-even) được dùng trực tiếp mà không định nghĩa ngắn gọn.
- **Đề xuất cách sửa cụ thể**:
  Chuẩn hóa toàn bộ logarit tự nhiên về `\ln`; thêm từ `positive` trước `normalized priors` tại `L697`; viết đầy đủ `\int_{\mathbb{R}^d}`, `\sum_{i=1}^{B}` và `\sum_{k\in\mathcal{S}_t^c}\sum_{i=1}^{n_k^c}`; viết tường minh $\mathcal{L}_{\mathrm{mix}} = \frac{1}{B}\sum_{i=1}^{B}\big[\xi\ell_{\mathrm{CE}}(h_k(r_i^{\mathrm{mix}}),y_i)+(1-\xi)\ell_{\mathrm{CE}}(h_k(r_i^{\mathrm{mix}}),y_{\pi(i)})\big]$; và giải thích ngắn gọn `stop-gradient operator $\operatorname{sg}[\cdot]$`, `$\operatorname{clip}(x,a,b)=\min(\max(x,a),b)$`, cùng `nearest-integer rounding $\operatorname{round}(\cdot)$`.

---

## Phần B — Thuật toán (Pseudocode), Đối chiếu Mã nguồn (`system/`) & Số liệu Thực nghiệm (`results/`, Tables/Figures) (16 Lỗi: `ALGO-01` – `EXP-06`)

> *Ghi chú khử trùng lặp mã định danh*: Để tránh tính trùng lỗi liên miền, `ALGO-06` (sai lệch công thức trong hình kiến trúc `framework.pdf`) đã được hợp nhất vào **`FIG-01`** (Phần C), và `CODE-04` (thiếu 7 hằng số ổn định/$\kappa=10$ và công thức biến thể chiếu theo lớp) đã được hợp nhất vào **`MATH-01`** và **`MATH-05`** (Phần A).

### `ALGO-01` — `Algorithm 1` Thiếu Đầu ra (`\ENSURE`), Thiếu Tập Dữ liệu Cục bộ & Khởi tạo Trọng số Chung $(\theta_f^{(0)}, \theta_h^{(0)})$, và Thiếu Bước Lưu Cache Trung bình Cục bộ $\mu_k^c$
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L586–L601` (Section III-A), `L1273–L1284` & `L1316–L1318` (Section III-F, `Algorithm 1`, bước 1–2 và 17); đối chiếu `system/main.py` (`L79, L98–L106, L184–L187`), `system/flcore/clients/clientbase.py` (`L20`), và `system/flcore/clients/clientscd.py` (`L34, L130, L180–L186`).
- **Trích dẫn nguyên văn**:
  ```latex
  \REQUIRE $K$ clients, $T$ rounds, $E$ local epochs;
  settings in Sections~\ref{sec:client}
  and~\ref{sec:server}.
  \STATE Initialize $\mathcal{C}_0=\emptyset$,
  zero moments and counts, and $\phi_0^c=\tau$.
  \STATE Initialize client counters $t_k=0$
  and local counts $n_k^c=0$.
  ...
  \STATE Recompute class-wise summaries from the
  statistics loader without gradients; upload.
  ```
- **Phân tích chi tiết**:
  1. **Thiếu dòng đầu ra `\ENSURE`**: `Algorithm 1` hoàn toàn không có `\ENSURE` (hoặc `Return`), mặc dù Mục III-A (`L586–L587`) xác định bài toán là học tập hợp các mô hình dự đoán cá nhân hóa $\{h_k \circ f_k\}_{k=1}^K$.
  2. **Thiếu dữ liệu đầu vào $\{\mathcal{D}_k\}_{k=1}^K$ và khởi tạo trọng số đồng nhất ban đầu**: Dòng `\REQUIRE` bỏ sót các tập dữ liệu riêng tư $\{\mathcal{D}_k\}_{k=1}^K$ và không khởi tạo tham số mạng $(\theta_{f_k}, \theta_{h_k}) \leftarrow (\theta_f^{(0)}, \theta_h^{(0)})$. Trong `main.py:98–106` và `clientbase.py:20` (`self.model = copy.deepcopy(args.model)`), **cả $K=20$ client đều được khởi tạo từ cùng một bản sao trọng số ban đầu giống hệt nhau** — đây chính là cơ chế nền tảng giúp các tọa độ đặc trưng xuất phát trong cùng hệ quy chiếu (`L599–L601`) dù không trao đổi trọng số mạng trong suốt quá trình huấn luyện.
  3. **Quên lưu cache $\mu_k^c$ cục bộ để phục vụ co cụm Empirical Bayes vòng kế tiếp**: Dòng 2 (`L1282–L1283`) chỉ khởi tạo $n_k^c = 0$ mà không khởi tạo vector trung bình lưu trữ $\mu_k^c = \mathbf{0}$; đồng thời Dòng 17 (`L1316–L1317`) chỉ viết `"upload"` mà không ghi rằng client $k$ còn phải **lưu lại $\{(n_k^c, \mu_k^c)\}$ vào bộ nhớ cục bộ** (`clientscd.py:130`: `self.local_stats = client_stats`) để tính mỏ neo co cụm `(\ref{eq:anchor})` ở lần tham gia tiếp theo của client đó.
- **Đề xuất cách sửa cụ thể**:
  ```latex
  \REQUIRE $K$ clients with private datasets $\{\mathcal{D}_k\}_{k=1}^K$, rounds $T$, local epochs $E$, shared initial parameters $(\theta_f^{(0)},\theta_h^{(0)})$; hyperparameters in Sections~\ref{sec:client} and~\ref{sec:server}.
  \ENSURE Personalized client models $\{(\theta_{f_k},\theta_{h_k})\}_{k=1}^K$.
  \STATE Server initializes $\mathcal{C}_0=\emptyset$, $(\mu_0^c,\sigma_0^c,n_0^c)=(\mathbf{0},\mathbf{0},0)$, and $\phi_0^c=\tau$ for $c\in\{1,\dots,C\}$.
  \STATE Each client $k$ sets $(\theta_{f_k},\theta_{h_k})\leftarrow(\theta_f^{(0)},\theta_h^{(0)})$, $t_k=0$, and cached local summaries $(\breve{n}_k^c,\breve{\mu}_k^c)\leftarrow(0,\mathbf{0})$ for $c\in\{1,\dots,C\}$.
  ...
  \STATE Extract $\{(n_k^c,\mu_k^c,\sigma_k^c)\}_{c=1}^C$ over $\mathcal{D}_{k,t}^{\mathrm{stat}}$ without gradients; update local cache $(\breve{n}_k^c,\breve{\mu}_k^c)\leftarrow(n_k^c,\mu_k^c)$ and upload $\{(n_k^c,\mu_k^c,\sigma_k^c)\}_{c=1}^C$ to the server.
  ```

---

### `ALGO-02` — Đặt Nhầm Phép Chiếu Mẫu `(\ref{eq:proj})` Ra Ngoài Vòng lặp Minibatch, Chưa Chỉ định Phạm vi Tham số Đạo hàm trong `Algorithm 1`, và Đếm Thiếu Lượt Truyền Thuận Thứ Hai Qua $h_k(z_i)$ ở Mục Độ Phức tạp
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L913–L923` (Section III-D), `L1301–L1315` (`Algorithm 1`, bước 9 và 13–16), và `L1351–L1353` (Section III-F, *Complexity*); đối chiếu `system/flcore/clients/clientscd.py` (`L150–L249, L280–L285, L446–L508`). *(Lưu ý: vấn đề 5 phương trình không nhãn `(6)–(10)` làm dịch số thứ tự phương trình được hợp nhất tại `LATEX-05` trong Phần C.)*
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1301-L1315 (Algorithm 1):
  \STATE Build fixed references
  using (\ref{eq:anchor})--(\ref{eq:proj}).
  ...
  \FOR{each minibatch over $E$ local epochs}
      \STATE Compute
      $\mathcal{L}_{\mathrm{head}}$
      (\ref{eq:head_loss}) and
      $\mathcal{L}_{\mathrm{enc}}$
      (\ref{eq:enc}).
      \STATE Compute separate head/extractor
      gradients, then step both optimizers.
  \ENDFOR

  % Overleaf/ieee2026.tex, L1351-L1353 (Complexity):
  With a linear classifier, real-feature, mixup,
  and virtual-feature head evaluations cost
  $\mathcal{O}((B+C)Cd)$.
  ```
- **Phân tích chi tiết**:
  1. **Không thể tính trước phép chiếu mẫu $\hat{z}$ ở ngoài vòng lặp minibatch**: Phương trình `(\ref{eq:proj})` (`Eq. (11)`, `L913–L923`) định nghĩa đồng thời cả mỏ neo chiếu $q^c = \operatorname{norm}_\epsilon((p^c - m)\odot w)$ lẫn vector đặc trưng mẫu đã chiếu $\hat{z} = \operatorname{norm}_\epsilon((z - m)\odot w)$. Vì $z_i = f_k(\mathrm{aug}(x_i))$ phụ thuộc vào từng minibatch cụ thể và thay đổi liên tục khi $\theta_{f_k}$ được cập nhật sau mỗi bước SGD, $\hat{z}_i$ **không thể** được tính cố định một lần tại bước 9 trước vòng lặp minibatch (`clientscd.py:150–249` chỉ tính trước $p^c, \widetilde{\sigma}^c, v^c, g^c, m, w, \phi^c$, còn `clientscd.py:281–284` chiếu $\hat{z}_i$ theo từng minibatch). Ngược lại, bên trong vòng lặp minibatch (`L1307–L1315`) lại bỏ sót hoàn toàn bước tăng cường ảnh $\mathrm{aug}(x_i)$, trích xuất $z_i = f_k(\mathrm{aug}(x_i))$ và chiếu $\hat{z}_i$.
  2. **Mơ hồ về phạm vi tham số lấy gradient (`L1313–L1314`)**: Hàm mục tiêu $\mathcal{L}_{\mathrm{enc}}$ (`Eq. (16)`, `L1028`) truyền trực tiếp $z_i$ (không tách gradient) đi qua đầu phân loại $h_k(z_i)$, nên $\mathcal{L}_{\mathrm{enc}}$ là hàm phụ thuộc đồng thời vào cả $\theta_{f_k}$ và $\theta_{h_k}$. Nếu chỉ viết chung chung `"Compute separate head/extractor gradients"` mà không ghi tường minh $g_h \leftarrow \nabla_{\theta_{h_k}}\mathcal{L}_{\mathrm{head}}$ và $g_f \leftarrow \nabla_{\theta_{f_k}}\mathcal{L}_{\mathrm{enc}}$ (tính trên $\theta_{h_k}$ trước khi bước tối ưu được thực hiện, đúng như `clientscd.py:498, 502`: `loss_head.backward(inputs=head_params)` và `loss_encoder.backward(inputs=enc_params)`), người cài đặt gọi `.backward()` thông thường của PyTorch sẽ làm dồn nhầm gradient từ $\mathcal{L}_{\mathrm{enc}}$ sang $\theta_{h_k}$.
  3. **Bỏ sót lượt truyền thuận thứ hai qua $h_k(z_i)$ trong phân tích độ phức tạp (`L1351–L1353`)**: Câu văn tại `L1351–L1353` chỉ liệt kê chi phí đánh giá tuyến tính của $h_k$ cho $\mathcal{L}_{\mathrm{head}}$ (*"real-feature, mixup, and virtual-feature head evaluations"*), bỏ quên việc $\mathcal{L}_{\mathrm{enc}}$ (`Eq. (16)`, `L1028`, tương ứng `clientscd.py:478`) thực thi thêm **một lượt truyền thuận riêng biệt thứ hai** $h_k(z_i)$ trên đặc trưng thật chưa tách gradient (cùng bậc tiệm cận $\mathcal{O}((B+C)Cd)$, nhưng chi phí nhân ma trận của nhánh đặc trưng thật gấp $2\times$: một lần trên $r_i = \operatorname{sg}[z_i]$ cho $\mathcal{L}_{\mathrm{head}}$ và một lần trên $z_i$ cho $\mathcal{L}_{\mathrm{enc}}$).
- **Đề xuất cách sửa cụ thể**:
  - Trong `Algorithm 1` (`L1301–L1315`), viết rõ các tham chiếu cố định và các bước bên trong minibatch:
    ```latex
    \STATE Precompute fixed references $\{p^c,\widetilde{\sigma}^c,v^c,g^c,q^c\}_{c\in\mathcal{A}_{t-1}}$ and $(m,w)$ via (\ref{eq:anchor})--(\ref{eq:proj}).
    ...
    \FOR{each minibatch $\mathcal{B}=\{(x_i,y_i)\}_{i=1}^{B}$ over $E$ local epochs}
        \STATE Extract $z_i = f_k(\mathrm{aug}(x_i))$; if $\mathbb{I}_{k,t}^{\mathrm{stat}}=1$, compute $\hat{z}_i$ via (\ref{eq:proj}) and sample virtual features via (\ref{eq:virt_feat}).
        \STATE Evaluate $\mathcal{L}_{\mathrm{head}}$ (\ref{eq:head_loss}) and $\mathcal{L}_{\mathrm{enc}}$ (\ref{eq:enc}).
        \STATE Compute $g_h \leftarrow \nabla_{\theta_{h_k}}\mathcal{L}_{\mathrm{head}}$ and $g_f \leftarrow \nabla_{\theta_{f_k}}\mathcal{L}_{\mathrm{enc}}$, then update $\theta_{h_k}$ and $\theta_{f_k}$.
    \ENDFOR
    ```
  - Tại `L1351–L1353`, bổ sung: `With a linear classifier, evaluating $h_k$ on un-detached real features (in $\mathcal{L}_{\mathrm{enc}}$) and on detached real, mixup, and virtual features (in $\mathcal{L}_{\mathrm{head}}$) costs $\mathcal{O}((B+C)Cd)$.`

---

### `ALGO-03` — Lỗi Khớp Cứng Cổng $\lambda > 0$ Làm Tắt Nhầm Hiệu chuẩn Đặc trưng Ảo của Head trong `Algorithm 1` (`L1294`) & `clientscd.py:412`, Cùng Lệnh Gọi Vô Điều kiện `Eqs. (16)–(17)` Gây `0/0` ở Vòng Khởi động Lạnh
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1108–L1109` (Section III-D) và `L1294–L1312` (Section III-F, `Algorithm 1`, bước 7–14); đối chiếu `system/flcore/clients/clientscd.py` (`L412–L418, L473–L474, L484–L488`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1294-L1312 (Algorithm 1):
  \IF{$\lambda>0$ \AND
  $t_k>T_{\mathrm{warm}}$ \AND
  $|\mathcal{A}_{t-1}|\geq2$}
      \STATE Set $\lambda_{k,t}=\lambda\min\{1,(t_k-T_{\mathrm{warm}})/T_{\mathrm{ramp}}\}$.
      \STATE Build fixed references using (\ref{eq:anchor})--(\ref{eq:proj}).
  \ELSE
      \STATE Set $\lambda_{k,t}=0$;
      disable statistical and virtual terms.
  \ENDIF
  \FOR{each minibatch over $E$ local epochs}
      \STATE Compute
      $\mathcal{L}_{\mathrm{head}}$
      (\ref{eq:head_loss}) and
      $\mathcal{L}_{\mathrm{enc}}$
      (\ref{eq:enc}).
  ```
- **Phân tích chi tiết**:
  1. **Mâu thuẫn cổng điều kiện $\lambda > 0$ giữa Section III-D và `Algorithm 1` / code**: Theo Section III-D (`L1028–L1036` và `L1090–L1109`), tham số $\lambda$ chỉ là trọng số của các số hạng chính quy hóa **bộ trích xuất đặc trưng** trong $\mathcal{L}_{\mathrm{enc}}$, trong khi hiệu chuẩn đặc trưng ảo cho **đầu phân loại** trong $\mathcal{L}_{\mathrm{head}}$ được điều khiển độc lập bởi $\lambda_{\mathrm{vl}}, \lambda_{\mathrm{vg}}$ và chỉ tắt khi đang warm-up hoặc thiếu mỏ neo toàn cục (`L1108–L1109`). Thế nhưng ở bước 7 của `Algorithm 1` (`L1294`) và trong `clientscd.py:412` (`if num_rounds < warmup_rounds or self.lamda_proto == 0.0: cache = None`), điều kiện tạo `cache` lại bị buộc chặt vào $\lambda > 0$. Hệ quả là bất kỳ cấu hình thử nghiệm/ablation nào đặt $\lambda = 0$ nhưng giữ $\lambda_{\mathrm{vl}}, \lambda_{\mathrm{vg}} > 0$ đều **vô tình làm tắt ngầm toàn bộ nhánh đặc trưng ảo của đầu phân loại**!
  2. **Gọi vô điều kiện `Eqs. (16)–(17)` ở bước 14 khi `cache` rỗng**: Ở vòng đầu tiên $t=1$, $\mathcal{C}_0 = \emptyset \implies |\mathcal{A}_0| = 0$ và $|\mathcal{I}| = 0$, nhánh `\ELSE` bỏ qua bước 9 nên các đại lượng $p^c, q^c, v^c, g^c, \widetilde{\sigma}^c$ hoàn toàn chưa tồn tại. Tuy nhiên bước 14 (`L1308–L1312`) vẫn gọi thẳng `(\ref{eq:head_loss})` (nơi xuất hiện phép chia cho $|\mathcal{I}|=0$ và $|\mathcal{A}|=0$) và `(\ref{eq:enc})` (nhân $\lambda_{k,t}=0$ với các hàm chưa xác định). Trong `clientscd.py:473, 484`, hai nhánh này thực tế được rẽ nhánh bỏ qua hoàn toàn khi `cache is None`.
- **Đề xuất cách sửa cụ thể**:
  Tách điều kiện khởi tạo tham chiếu thống kê $\mathbb{I}_{k,t}^{\mathrm{stat}} = \mathbb{I}(t_k > T_{\mathrm{warm}} \wedge |\mathcal{A}_{t-1}| \ge 2)$ ra khỏi điều kiện $\lambda > 0$ cả trong `Algorithm 1` (`L1294`) lẫn `clientscd.py:412` (chỉ xây `cache` khi $t_k > T_{\mathrm{warm}}$, $|\mathcal{A}_{t-1}| \ge 2$, và $\lambda + \lambda_{\mathrm{vl}} + \lambda_{\mathrm{vg}} > 0$); hoặc nếu giữ nguyên hành vi hiện tại của code, phải ghi rõ tại `L1108–L1109` rằng các số hạng đặc trưng ảo cũng bị tắt khi $\lambda = 0$. Đồng thời bảo vệ `Eq. (17)` bằng hàm chỉ báo $\mathbb{I}_{k,t}^{\mathrm{stat}}$ như đề xuất ở `MATH-04`.

---

### `ALGO-04` — Thiếu Điều kiện Bảo vệ Lớp Hoạt động ($\bar{n}_t^c > 0$) ở Bước Gom Moment Server trong `Algorithm 1` (`L1320–L1325`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1140–L1165` (Section III-E) và `L1320–L1325` (Section III-F, `Algorithm 1`, bước 19); đối chiếu `system/flcore/servers/serverscd.py` (`L189–L206`).
- **Trích dẫn nguyên văn**:
  ```latex
  \STATE Pool $(\bar{\mu}_t^c,\bar{v}_t^c)$
  using (\ref{eq:server_mean})--(\ref{eq:server_var});
  set
  $\bar{\sigma}_t^c
  =[\max(\bar{v}_t^c,\epsilon_v\mathbf{1})]^{\odot1/2}$.
  ```
- **Phân tích chi tiết**:
  Các phương trình gom moment phía server `(\ref{eq:server_mean})--(\ref{eq:server_var})` (`Eqs. (18)–(19)`, `L1141–L1165`) có phép chia $1/\bar{n}_t^c$ với $\bar{n}_t^c = \sum_{k\in\mathcal{S}_t^c} n_k^c$ và chỉ xác định cho các lớp có $\bar{n}_t^c > 0$. Dưới lệch nhãn mạnh (ví dụ $C=102$ trên Flowers102 hoặc $C=200$ trên Tiny-ImageNet ở $\alpha=0.1$, do loại bỏ minibatch cuối không đủ 32 mẫu nên một lớp ít mẫu trên toàn hệ thống có thể có $\bar{n}_t^c = 0$) hoặc khi tham gia một phần (partial participation), sẽ tồn tại các lớp có $\bar{n}_t^c = 0$ tại vòng $t$. Mã nguồn `serverscd.py:190` kiểm tra chặt `if batch_count > 0:`, nhưng bước 19 của `Algorithm 1` (`L1320–L1325`) lại gọi `(\ref{eq:server_mean})--(\ref{eq:server_var})` không có điều kiện lọc lớp, dẫn tới phép chia $0/0$ khi $\bar{n}_t^c = 0$.
- **Đề xuất cách sửa cụ thể**:
  Sửa bước 19 của `Algorithm 1` (`L1320–L1325`, đồng thời tách dòng để triệt tiêu lỗi `Underfull \hbox` ở `LATEX-07`):
  ```latex
  \STATE For each class $c$ with $\bar{n}_t^c > 0$, pool $(\bar{\mu}_t^c,\bar{v}_t^c)$ via (\ref{eq:server_mean})--(\ref{eq:server_var}) and compute $\bar{\sigma}_t^c$ via (\ref{eq:server_std}).
  ```

---

### `ALGO-05` — Mâu thuẫn Nội bộ & Giữa Bài báo với Code về Dung lượng Gói Tin Phát sóng Downlink ($C(2d+2)$ Có Chứa $n_t^c$ Không Dùng Đến vs. Bộ Ba $(\mu^c,\sigma^c,\phi^c)$ Không Chứa $n_t^c$)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L472–L473`, `L768–L770`, `L1260`, `L1330–L1341`, `L1464`, `L1535`; đối chiếu `system/flcore/clients/clientscd.py` (`L133–L146`) và `system/flcore/servers/serverscd.py` (`L225–L272`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L768-L770 (Section III-C):
  uses the latest global statistics $\{(\mu_{t-1}^c,\sigma_{t-1}^c,\phi_{t-1}^c)\}_{c\in\mathcal{C}_{t-1}}$,

  % Overleaf/ieee2026.tex, L1330-L1341 (Algorithm 1 & Complexity):
  \STATE Broadcast $\{(\mu_t^c,\sigma_t^c,n_t^c,\phi_t^c)\}_{c=1}^{C}$ to all clients.
  ...
  A complete client upload contains $C(2d+1)$ scalar values for class means, standard deviations, and counts. The server broadcasts $C(2d+2)$ values per recipient, including class temperatures.
  ```
- **Phân tích chi tiết**:
  1. **Mâu thuẫn nội bộ giữa các mục trong bài báo**: Mục III-C (`L768–L770`), Mục III-D (`L825`) và cuối Mục III-E (`L1260`) nói rằng phía client nhận/sử dụng bộ 3 thành phần $(\mu^c, \sigma^c, \phi^c)$ (tương ứng $C(2d+1)$ giá trị vô hướng nếu truyền đủ $C$ lớp). Ngược lại, Mục II-C (`L472`), `Algorithm 1` bước 22 (`L1331`), phần Độ phức tạp (`L1340`), Mục IV (`L1464`) và Mục V-A (`L1535`) lại tính gói tin downlink là bộ 4 thành phần $(\mu_t^c, \sigma_t^c, n_t^c, \phi_t^c)$ gồm $C(2d+2)$ giá trị vô hướng — và giải thích mức tăng từ $C(2d+1)$ lên $C(2d+2)$ là do *"including class temperatures"*, điều chỉ đúng đại số khi server gửi kèm cả số đếm toàn cục đã làm trơn $n_t^c$.
  2. **Đối chiếu mã nguồn**: Trong `serverscd.py:255–272`, dictionary `global_stats` phát xuống client có chứa khóa `'count'` ($n_t^c$), nhưng hàm `receive_global_stats` tại `clientscd.py:133–146` **chỉ đọc `'mean'`, `'std'` và `'phi'`, bỏ qua hoàn toàn `'count'`** (vì phương trình co cụm `(\ref{eq:anchor})` ở client chỉ dùng số đếm cục bộ $n_k^c$, còn $n_t^c$ chỉ cần ở phía server cho `Eqs. (20)–(21)`).
- **Đề xuất cách sửa cụ thể**:
  Chọn 1 trong 2 phương án nhất quán xuyên suốt toàn bài:
  - **Phương án 1 (Tối ưu giao thức — Khuyên dùng)**: Lược bỏ $n_t^c$ dư thừa khỏi gói phát sóng downlink trong `serverscd.py` và tại `L472, L1331, L1340, L1464, L1535`, giúp cả uplink `{(n_k^c, \mu_k^c, \sigma_k^c)}` lẫn downlink `{(\mu_t^c, \sigma_t^c, \phi_t^c)}` đối xứng hoàn hảo ở mức đúng **$C(2d+1)$ vô hướng**.
  - **Phương án 2 (Giữ nguyên code hiện tại)**: Thống nhất viết bộ 4 thành phần $\{(\mu_{t-1}^c, \sigma_{t-1}^c, n_{t-1}^c, \phi_{t-1}^c)\}$ ở `L769` và ghi chú ngắn gọn tại `L1340` rằng $n_t^c$ nằm trong từ điển trạng thái phát sóng ($C(2d+2)$ vô hướng) dù phía client chỉ tiêu thụ $(\mu_t^c, \sigma_t^c, \phi_t^c)$.

---

### `CODE-01` — Vòng lặp Lệch Một Vòng (`101` vs. `100` Rounds) VÀ Bất Đối xứng Thời điểm Đánh giá (`Post-Train` vs. `Pre-Train` `evaluate()`) Giữa `FedSCD` và Cả 6 Baseline (Làm Lệch Ngang Toàn bộ Trục Hoành `0..100` của `Fig. 1` và `Fig. 3`)
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L199–L209` (`Fig. 1`), `L1285` (`Algorithm 1`), `L1453` (Section IV), `L1516–L1532` (`Fig. 3`, Section V-A); đối chiếu `system/flcore/servers/serverscd.py` (`L48–L77`), `servertgp.py` (`L80–L94`), `servercof.py` (`L49–L69`), `serversa.py` (`L65–L79`), `serverorgp.py` (`L110–L124`), `serverplvm.py` (`L63–L84`), `servergmkd.py` (`L82–L96`), và `Figure_Plot/test_acc.csv` (`L2, L102`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1453:
  All methods are trained for $T{=}100$ communication rounds with full client participation, one local epoch per round ($E{=}1$)...
  ```
  Đối chiếu với `system/flcore/servers/serverscd.py` (`L48–L76`):
  ```python
  for i in range(self.global_rounds + 1):
      ...
      for client in self.selected_clients:
          client.train()
      self.receive_protos()
      self.global_stats = proto_aggregation_scd(...)
      self.send_protos()
      if i % self.eval_gap == 0:
          self.evaluate()
  ```
  Trong khi tất cả 6 server baseline (ví dụ `servertgp.py:80–94`, `servercof.py:49–69`) đều gọi `self.evaluate()` **trước** `client.train()`.
- **Phân tích chi tiết**:
  1. **`FedSCD` được huấn luyện 101 vòng thay vì 100 vòng**: Khi truyền `--global_rounds 100`, vòng lặp `range(self.global_rounds + 1)` trong `serverscd.py` chạy qua `i = 0, 1, ..., 100` (**tổng cộng 101 vòng cập nhật SGD cục bộ**) mà không có câu lệnh `break`. Ngược lại, `FedCOF` và `FedPLVM` dừng bằng `break` tại `i == 100` ngay sau khi đánh giá mô hình sau 100 vòng train (`servercof.py:60–61`, `serverplvm.py:74–75`); còn `FedTGP`, `FedSA`, `FedORGP`, `FedGMKD` gọi `self.evaluate()` lần cuối ở đầu `i = 100` **trước khi** chạy lượt `client.train()` thứ 101, nên mọi chỉ số được ghi log của cả 6 baseline chỉ phản ánh tối đa **100 vòng huấn luyện**, còn `FedSCD` hưởng trọn **101 vòng huấn luyện**!
  2. **Lệch pha ngang 1 vòng và trục hoành 101 điểm (`0..100`) trên `Fig. 1` và `Fig. 3`**: Vì `serverscd.py` gọi `evaluate()` *sau* `client.train()` trong khi cả 6 baseline gọi `evaluate()` *trước* `client.train()`, tại điểm hoành độ `Round = 0` trong `Figure_Plot/test_acc.csv` (`L2`), `prototype_margin.csv` (`L2`), `bhattacharyya_overlap.csv` (`L2`), `Fig. 1` và `Fig. 3`, **`FedSCD` thực chất đã hoàn thành 1 vòng huấn luyện đầy đủ** (đạt ngay `41.00%` test accuracy, margin `2.06`, overlap `0.11` trên CIFAR-10 $\alpha=0.5$ ở `Fig. 1`; và vọt lên `~49%` ngay tại `Round 0` trên CIFAR-10 $\alpha=0.1$ ở `Fig. 3`), trong khi **cả 6 baseline tại `Round 0` đang đánh giá trên trọng số khởi tạo ngẫu nhiên chưa huấn luyện** (`8.02%–9.97%` accuracy, margin `0.08`, overlap `0.74`)! Nói cách khác, tại mọi hoành độ $t \in \{0, \dots, 100\}$ trên `Fig. 1` (gồm 101 mốc đánh giá thay vì $t=1,\dots,100$ như `Algorithm 1` `L1285`), đường cong của `FedSCD` sau $t+1$ vòng đang được vẽ đè lên đường cong của các baseline sau $t$ vòng.
- **Đề xuất cách sửa cụ thể**:
  - **Sửa trong mã nguồn chuẩn (`serverscd.py:48–76`)** để đồng bộ tuyệt đối với `servercof.py:49–62`:
    ```python
    for i in range(self.global_rounds + 1):
        self.selected_clients = self.select_clients()
        if i % self.eval_gap == 0:
            self.evaluate()
        if i == self.global_rounds:
            break
        for client in self.selected_clients:
            client.train()
        self.receive_protos()
        ...
    ```
  - **Nếu chưa chạy lại thực nghiệm ngay**, bắt buộc phải công bố minh bạch sự lệch chỉ số đánh giá này trong Section IV (*Metrics*, `L1458–L1464`) và trong caption của `Fig. 1` (`L207`) cùng `Fig. 3` (`L1519`), hoặc dịch trục hoành của `FedSCD` sang `t = 1..101` (cắt ở `t = 100`) trong script vẽ hình `Figure_Plot/plot_combined_3panels.py` và `plot_4datasets_test_acc.py`.

---

### `CODE-02` — Năm Siêu tham số Cốt lõi trong Bài báo (`L1453`) Mâu thuẫn Hoàn toàn với Toàn bộ Mặc định Mã nguồn, Script Benchmark và `ABLATION_RUN_GUIDE.md`
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1453` (Section IV) & `L1624` (Section V-C); đối chiếu `system/main.py` (`L308, L334, L336, L344, L346`), `system/flcore/clients/clientscd.py` (`L23, L39, L53, L55–L59`), `system/flcore/servers/serverscd.py` (`L32`), `system/run_target_experiments.py` (`L24–L35`), `system/run_target_experiments.ps1` (`L153–L163`), `system/run_benchmark_matrix.py` (`L73–L83`), `ABLATION_RUN_GUIDE.md` (`L28, L86–L94`), `BENCHMARK_RUN_GUIDE.md` (`L136, L160`), `Figure_Plot/plot_combined_3panels.py` (`L17`), `Figure_Plot/plot_4datasets_test_acc.py` (`L6`), và `Figure_Plot/wandb_seed2006_max_acc.csv` (`L62–L73`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1453:
  We set the base temperature $\tau{=}0.1$, temperature modulation margin $\rho{=}0.1$, server EMA coefficient $\beta{=}0.6$, representation regularization weight $\lambda{=}0.3$, instance contrastive weight $\lambda_{\text{ins}}{=}0.75$, warm-up length $T_{\text{warm}}{=}1$, ramp-up length $T_{\text{ramp}}{=}5$, and virtual-feature calibration weights $\lambda_{\text{vl}}{=}0.3$ and $\lambda_{\text{vg}}{=}0.3$.
  ```
  Đối chiếu `ABLATION_RUN_GUIDE.md` (`L28`), `run_target_experiments.py` (`L25–L34`) và `main.py` (`L308–L347`):
  ```text
  -lam 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5   (và --lamda_virt_local mặc định = 0.0)
  ```
- **Phân tích chi tiết**:
  Năm siêu tham số điều khiển trực tiếp hàm mất mát và làm trơn server có mức chênh lệch từ **$3\times$ đến $15\times$** (hoặc bật so với tắt hoàn toàn) giữa bản thảo `ieee2026.tex` và toàn bộ kho mã nguồn hiện tại:

  | Siêu tham số | Cờ CLI trong `system/` | Giá trị trong `ieee2026.tex` (`L1453`) | Giá trị trong `main.py`, scripts & guides | Mức sai lệch |
  |:---|:---|:---:|:---:|:---|
  | Hệ số làm trơn EMA server $\beta$ | `--proto_momentum` | **`0.6`** | **`0.1`** | Gấp **$6\times$** |
  | Trọng số chính quy hóa encoder $\lambda$ | `-lam` / `-lp` | **`0.3`** | **`0.1`** | Gấp **$3\times$** |
  | Trọng số tương phản lai $\lambda_{\mathrm{ins}}$ | `--lamda_ins` | **`0.75`** | **`0.05`** | Gấp **$15\times$** |
  | Trọng số đặc trưng ảo cục bộ $\lambda_{\mathrm{vl}}$ | `--lamda_virt_local` | **`0.3`** | **`0.0`** *(tắt hẳn)* | Đang **tắt** trong code |
  | Trọng số đặc trưng ảo toàn cục $\lambda_{\mathrm{vg}}$ | `--lamda_virt_global` | **`0.3`** | **`0.05`** | Gấp **$6\times$** |

  Hệ quả nghiêm trọng về khả năng tái lập khoa học: nếu người đọc hoặc phản biện chạy đúng các script được lưu trong repo (`run_target_experiments.py` / `ABLATION_RUN_GUIDE.md`), kết quả thu được chính là bộ số cũ trong `Figure_Plot/wandb_seed2006_max_acc.csv` (`L62–L73`: CIFAR-10 `90.71 / 74.61 / 69.17`, CIFAR-100 `55.67 / 35.72 / 29.37`, Flowers102 `58.69 / 32.96 / 25.01`, Tiny-ImageNet `36.77 / 18.08 / 14.28` — nơi `FedSCD` **thua `FedCOF` ở 6/12 cấu hình**!), chứ **không thể tái tạo được** con số đã tinh chỉnh `76.66%` và chuỗi thắng 12/12 cấu hình trong `Tables I–III`. Ngoài ra, `Figure_Plot/plot_combined_3panels.py:17` vẫn đang gán cứng WandB run ID cũ `"ngyaoosa"` (`74.61%`), còn `Figure_Plot/plot_4datasets_test_acc.py:6` trỏ cứng tới đường dẫn ngoài repo `/Users/phucvu/Downloads/wandb_export_cifar10_adjusted.csv`.
- **Đề xuất cách sửa cụ thể**:
  Kiểm tra lại chính xác log WandB của đợt chạy tinh chỉnh tạo ra bộ kết quả `91.63 / 76.66 / 70.94` trong `Tables I–III`, sau đó **đồng bộ hóa 100%** giữa `Overleaf/ieee2026.tex` (`L1453, L1624`) và các file mặc định/kịch bản trong `system/` (`main.py`, `clientscd.py`, `serverscd.py`, `run_target_experiments.py`, `run_benchmark_matrix.py`, `ABLATION_RUN_GUIDE.md`, `BENCHMARK_RUN_GUIDE.md`), đồng thời cập nhật run ID mới trong `plot_combined_3panels.py` và đưa file CSV dữ liệu vẽ hình vào trong thư mục `Figure_Plot/`.

---

### `CODE-03` — Công thức Hằng số Co cụm Empirical Bayes $n_0 = \max(5, |\mathcal{D}_k|/(2C))$ trong Bài báo (`L1454`) Mâu thuẫn với Hằng số Cứng `50.0` trong `clientscd.py:185`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L844–L848` (`Eq. (5)`) và `L1453–L1454` (Section IV); đối chiếu `system/flcore/clients/clientscd.py` (`L185`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1453-L1454:
  For anchor shrinkage and dispersion gating, we set $\gamma_{\max}{=}0.25$, $n_{0}{=}\max(5, |\mathcal{D}_k|/(2C))$, $\alpha_v{=}0.25$, $\alpha_q{=}10^{-2}$...
  ```
  Đối chiếu `system/flcore/clients/clientscd.py` (`L185`):
  ```python
  gamma_c = self.bayes_shrink_max * (n_c / (n_c + 50.0))
  ```
- **Phân tích chi tiết**:
  1. Trong `clientscd.py:185`, $n_0$ được gán cứng bằng hằng số bất biến **`50.0`** cho mọi client và mọi tập dữ liệu, hoàn toàn **không phải** công thức thích ứng $\max(5, |\mathcal{D}_k|/(2C))$.
  2. Trên 4 tập dữ liệu thực nghiệm ($K=20$, chia 75% tập train), giá trị trung bình của $\max(5, |\mathcal{D}_k|/(2C))$ là **`112.5`** trên CIFAR-10 (lớn gấp $2.25\times$ so với `50.0`), **`11.25`** trên CIFAR-100 (nhỏ hơn $4.4\times$), **`5.0`** trên Flowers102 (nhỏ hơn $10\times$), và **`10.3`** trên Tiny-ImageNet (nhỏ hơn $4.8\times$). Hơn nữa, nếu $n_0$ thực sự phụ thuộc vào $|\mathcal{D}_k|$ thì trong phương trình `(5)` (`L847`) phải viết có chỉ số client là $n_{0,k}$ chứ không thể viết $n_0$ không chỉ số.
- **Đề xuất cách sửa cụ thể**:
  Nếu các lượt chạy thực nghiệm dùng code `clientscd.py:185` với `50.0`, sửa `L1454` trong `Overleaf/ieee2026.tex` từ `$n_{0}{=}\max(5, |\mathcal{D}_k|/(2C))$` thành `$n_{0}{=}50$`. Ngược lại, nếu nhánh code chạy thực nghiệm mới đã đổi sang công thức $\max(5, |\mathcal{D}_k|/(2C))$, cần commit phiên bản `clientscd.py` đó vào repo và đổi ký hiệu ở `Eq. (5)` thành $n_{0,k}$.

---

### `CODE-05` — Bỏ sót Hàm Kích hoạt `ReLU` Sau Lớp `fc1` (Làm Không gian Đặc trưng Thuộc Góc Phần tư Không Âm $\mathbb{R}_{\ge 0}^d$), `DataLoader` Không Xáo trộn (`shuffle=False`, `drop_last=True`), Crop Dùng Chung Toàn Batch và Chi tiết Tiền xử lý Tập dữ liệu
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L578–L580` (Section III-A), `L1078–L1088` (Section III-D), `L1443–L1456` (Section IV); đối chiếu `system/flcore/trainmodel/models.py` (`L211–L246`), `system/flcore/clients/clientbase.py` (`L64–L65`), `system/flcore/clients/clientscd.py` (`L81–L83, L353`), và `scripts/prepare_all_datasets.py` (`L165–L216, L240–L275`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1447:
  All clients use the same CNN architecture comprising two $5{\times}5$ convolutional layers followed by a fully connected layer producing a $d{=}512$-dimensional feature representation, paired with a linear classifier head.

  % Overleaf/ieee2026.tex, L1456:
  while virtual features are clipped coordinate-wise at zero whenever all usable anchor coordinates are at least $-10^{-6}$.
  ```
- **Phân tích chi tiết**:
  1. **Thiếu `ReLU` sau lớp `fc1` khiến câu mô tả cắt ngưỡng $\Pi(\cdot)$ trở nên khó hiểu**: Trong `models.py:214–238`, kiến trúc `FedAvgCNN` thực thi chuỗi `Conv2d(3,32,5) -> ReLU -> MaxPool2d(2) -> Conv2d(32,64,5) -> ReLU -> MaxPool2d(2) -> Linear(dim,512) -> ReLU`. Vì đầu ra của `fc1` đi qua `ReLU`, mọi vector đặc trưng thực $z = f_k(x)$ đều nằm trên hình nón không âm $\mathbb{R}_{\ge 0}^d$ chứ không trải khắp $\mathbb{R}^d$ (`L578`). Việc bỏ sót chi tiết `ReLU` cuối encoder làm người đọc không hiểu vì sao xuất hiện các chiều chết ($p_j^c = 0, \sigma_j^c = 0$), vì sao bắt buộc phải dịch tâm bởi $m$ trước khi chuẩn hóa $\ell_2$ ở `Eq. (11)`, và vì sao `L1456` lại đặt quy tắc kiểm tra $p_j^c \ge -10^{-6}$ để cắt không âm $\Pi(a) = \max(a, \mathbf{0})$ cho đặc trưng ảo (`clientscd.py:353`).
  2. **`DataLoader(..., drop_last=True, shuffle=False)` loại bỏ vĩnh viễn mẫu dư trong suốt 100 vòng**: Tại `clientbase.py:64–65`, cờ mặc định là `shuffle=False` và không có bất kỳ script nào truyền `--shuffle`. Khi kết hợp `shuffle=False` với `drop_last=True` ($B=32$), `n_k mod 32` mẫu cuối cùng của mỗi client **bị loại bỏ vĩnh viễn khỏi cả quá trình huấn luyện lẫn trích xuất thống kê trong toàn bộ 100 vòng** (chứ không chỉ bị loại ở một lượt chia ngẫu nhiên như người đọc hiểu từ `L1455`), đồng thời $\mathcal{L}_{\mathrm{SC}}$ luôn đối chiếu cố định cùng các nhóm 32 mẫu qua mọi vòng. Ngoài ra, hàm `_apply_scd_view` (`clientscd.py:81–83`) lấy mẫu duy nhất một cặp tọa độ cắt `(top, left)` dùng chung cho toàn bộ 32 ảnh trong minibatch thay vì cắt độc lập từng ảnh.
  3. **Thiếu chi tiết gộp split và resize ảnh đầu vào**: Script `prepare_all_datasets.py` gộp chung tập `train + test` (và `+ val` với Flowers102, tổng 8.189 ảnh được resize về $64\times 64$; Tiny-ImageNet gộp `train + val` gồm 110.000 ảnh $64\times 64$) trước khi chia Dirichlet cho $K=20$ client rồi tách 75%/25% tại từng client.
- **Đề xuất cách sửa cụ thể**:
  Tại Section IV (`L1443–L1456`), mô tả đầy đủ: (i) kiến trúc CNN gồm hai khối `Conv(5x5)-ReLU-MaxPool(2x2)` nối tiếp bởi `Linear-ReLU` tạo biểu diễn không âm $z \in \mathbb{R}_{\ge 0}^{512}$ (giải thích trực tiếp cho phép chiếu cắt không âm $\Pi(a)=\max(a,\mathbf{0})$); (ii) kích thước ảnh $32\times 32$ trên CIFAR-10/100 và $64\times 64$ trên Flowers102/Tiny-ImageNet cùng quy trình gộp tập gốc trước khi phân hoạch $\mathrm{Dir}(\alpha)$; và (iii) làm rõ việc dùng minibatch đầy đủ cố định (`drop_last=True`, `shuffle=False`) hoặc bật `shuffle=True` khi chạy lại thực nghiệm.

---

### `CODE-06` — Lỗi Logic Ghi đè Cờ CLI `-lp 1.0` trong `clientscd.py:41–45` và Bất Đối xứng Cách Trích xuất Thống kê Giữa Các Thuật toán trong `motivation_metrics.py`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `system/flcore/clients/clientscd.py` (`L41–L45`), `system/utils/motivation_metrics.py` (`L18–L67`), `system/flcore/clients/clientsa.py` (`L186–L208`), đối chiếu `Overleaf/ieee2026.tex` (`L1464, L1549–L1552`).
- **Trích dẫn nguyên văn**:
  ```python
  # system/flcore/clients/clientscd.py, L41-L45:
  v_proto = getattr(args, 'lamda_proto', None)
  if v_proto is not None and float(v_proto) != 1.0:
      self.lamda_proto = float(v_proto)
  else:
      self.lamda_proto = self.lamda
  ```
- **Phân tích chi tiết**:
  1. **Bẫy tham số dòng lệnh `-lp 1.0`**: Vì `--lamda_proto` (`-lp`) có giá trị mặc định là `1.0` trong `main.py:314`, đoạn kiểm tra `float(v_proto) != 1.0` tại `clientscd.py:42` coi `1.0` là "chưa được truyền", khiến cờ `-lp` ghi đè lên `-lam` ở mọi giá trị thực **ngoại trừ đúng giá trị `1.0`** (nếu người dùng muốn sweep ablation với `-lp 1.0`, code sẽ âm thầm quay về giá trị của `-lam`!).
  2. **Khác biệt chế độ trích xuất thống kê khi vẽ đường cong chồng lấn `Fig. 1(b)`**: Trong `motivation_metrics.py:18–67`, `FedSCD` và `FedTGP` cung cấp thống kê từ lượt chạy `eval()` sau huấn luyện (`drop_last=True`, lớp đơn mẫu `std=1e-6`); riêng **`FedSA` (`clientsa.py:186–208`) tích lũy thống kê trực tuyến ngay trong chế độ `train()` khi trọng số đang thay đổi qua từng minibatch** (làm phồng phương sai nội client của `FedSA`); còn `FedCOF`, `FedORGP` và `FedGMKD` dùng nhánh dự phòng `eval()` trên toàn bộ tập train cục bộ (`drop_last=False`, lớp đơn mẫu `std=0.0`).
- **Đề xuất cách sửa cụ thể**:
  Chú thích ngắn gọn sự khác biệt nguồn thống kê chẩn đoán này tại Section V-B (`L1550–L1551`, nơi bài báo đã nhắc tới *"statistics extraction"* nhưng chưa nói cụ thể), và khi dọn dẹp repo code hãy đặt `default=None` cho `--lamda_proto` trong `main.py:314`.

---

### `EXP-01` — Không Công bố Quy tắc Lấy Độ Chính xác Đỉnh (`max_acc`) trong `Table I`, Hai Lần Tham chiếu Treo Tới *"Section IV Limitations"* Không Tồn tại, và Che khuất Việc `FedTGP` Sụp đổ / `FedPLVM` Crash Giữa Chừng
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1441–L1464` & `L1466–L1490` (Section IV & `Table I`), `L1500`, `L1510–L1511` & `L1530–L1532` (Section V-A), `L1641` (Section V-C), và `L1657` (Section VI); đối chiếu `Figure_Plot/wandb_seed2006_max_acc.csv` (`L2–L77`) và `results/runs_results/*.csv` / `runs_results.xlsx`.
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1458-L1464 (Section IV):
  Predictive performance is measured by the weighted local test accuracy $A_{\mathrm{local}} = \dots$, reported as single-run results in Table~\ref{tab:com}.

  % Overleaf/ieee2026.tex, L1500 & L1531-L1532 (Section V-A):
  Table~\ref{tab:com} summarizes the available accuracy point estimates for FedSCD and six baseline implementations, subject to the comparison-scope limitations in Section IV.
  ...
  and their cross-method interpretation inherits the checkpoint and update-budget limitations in Section IV.

  % Overleaf/ieee2026.tex, L1641 & L1657:
  Further validation should reconcile the run-level evaluation protocol, report repeated-run uncertainty...
  ... with repeated-run uncertainty and matched-budget validation still unavailable.
  ```
- **Phân tích chi tiết**:
  1. **Không tiết lộ rằng `Table I` báo cáo giá trị cực đại lịch sử (`max_acc`) thay vì vòng cuối**: Section IV (`L1458–L1464`) không hề nói độ chính xác trong `Table I` được lấy ở vòng truyền thông nào. Đối chiếu trực tiếp `Figure_Plot/wandb_seed2006_max_acc.csv` và `results/runs_results/runs_results.xlsx` (Sheet `Summary`, dòng 2: *"Best Test Accuracy (%) [Peak Performance across 100 Rounds]"*) chứng minh toàn bộ `Table I` đang báo cáo **độ chính xác kiểm thử cao nhất qua mọi vòng đánh giá (`max_acc = max_t A_local^(t)`)**, tức chọn checkpoint tối ưu trực tiếp trên tập test mà không có tập validation.
  2. **Tham chiếu đứt gãy nghiêm trọng tới những "giới hạn ở Section IV" hoàn toàn không tồn tại**: Tại `L1500` và `L1531–L1532`, văn bản hai lần dẫn người đọc quay lại *"the comparison-scope limitations in Section IV"* và *"the checkpoint and update-budget limitations in Section IV"*, đồng thời tại `L1510–L1511, L1641, L1657` liên tục ám chỉ rằng ngân sách cập nhật (update budget) hoặc giao thức chọn checkpoint giữa các thuật toán chưa đồng nhất. Thế nhưng **đọc toàn bộ Section IV (`L1441–L1464`) hoàn toàn không có bất kỳ câu chữ nào bàn về checkpoint selection, update-budget mismatch hay comparison-scope limitations** (ngược lại, `L1453` khẳng định mọi phương pháp đều chạy chung $T=100, E=1, B=32, \mathrm{lr}=0.01$)! Đây là dấu vết cắt xóa dở dang gây mâu thuẫn nội tại chí mạng, khiến phản biện lập tức đặt nghi vấn về tính minh bạch của toàn bộ thực nghiệm.
  3. **Che khuất sự sụp đổ phân kỳ của `FedTGP` và việc `10/12` lượt chạy `FedPLVM` bị crash giữa chừng**:
     - **`FedTGP` sụp đổ cuối quá trình huấn luyện**: Trên Flowers102 ($\alpha = 0.1, 0.5, 1.0$), `Table I` in các giá trị đỉnh giữa kỳ `54.79%`, `31.26%`, `24.48%` (đạt ở vòng 57, 38 và 36), trong khi độ chính xác cuối kỳ sụp đổ thảm hại xuống **`0.58%`**, **`0.58%`** (`state=failed` tại bước 80), và **`0.54%`**! Trên Tiny-ImageNet ($\alpha = 0.1, 0.5, 1.0$), `FedTGP` cũng suy giảm mạnh từ đỉnh `32.49%`, `16.64%`, `13.52%` xuống còn **`20.93%`**, **`9.34%`**, **`7.06%`** ở vòng 100.
     - **`FedPLVM` bị lỗi dừng sớm ở 10 trên 12 cấu hình**: Kiểm tra `wandb_seed2006_max_acc.csv` cho thấy **10/12 lượt chạy `FedPLVM` không hoàn thành đủ 100 vòng**: 9 lượt có trạng thái `state=failed` (điển hình Flowers102 $\alpha=0.1$ bị crash chỉ sau **3 vòng** với `len=3`, CIFAR-100 $\alpha=0.5$ dừng ở vòng 60), và lượt Tiny-ImageNet $\alpha=1.0$ (`7.64%`) được trích xuất khi vẫn đang ở trạng thái `state=running`!
- **Đề xuất cách sửa cụ thể**:
  - Khai báo trung thực và tường minh ngay tại Section IV (*Metrics*, `L1458–L1464`) quy tắc trích xuất số liệu của `Table I` (ví dụ: báo cáo song song **Final-Round Accuracy** hoặc trung bình 5 vòng cuối cùng bên cạnh **Peak Accuracy** $\max_t A_{\mathrm{local}}^{(t)}$, kèm ký hiệu chú thích chân bảng `^\dagger` cho các lượt chạy baseline bị phân kỳ `NaN`/dừng sớm trước vòng 100).
  - Xóa bỏ các tham chiếu treo tới *"limitations in Section IV"* ở `L1500, L1531–L1532` sau khi đã trình bày rõ ràng, trực diện mọi chi tiết về giao thức đánh giá và chi phí hai nhánh tối ưu tách rời (`Section III-F`) ngay trong Section IV/V.

---

### `EXP-02` — Chỉ Riêng `FedSCD` Được Bật Augmentation Ảnh Trên GPU (`_apply_scd_view`) và Head Mixup (Cả 6 Baseline Đều Tắt), Gây Nghịch lý Mô hình Cục bộ Không Giao tiếp ở `Table II` (`72.79%`) Thắng Toàn bộ 6 Thuật toán FL ở `Table I` (`\le 71.70%`)
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1455–L1456` (Section IV), `L1479–L1486` (`Table I`), `L1570–L1583` (`Table II`); đối chiếu `system/flcore/clients/clientscd.py` (`L62–L84, L446, L460–L470`) và toàn bộ 6 file baseline client (`clienttgp.py`, `clientplvm.py`, `clientgmkd.py`, `clientsa.py`, `clientorgp.py`, `clientcof.py`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1455-L1456:
  Data augmentation applies random horizontal flips ($p{=}0.5$) and random cropping after four-pixel reflection padding.

  % Overleaf/ieee2026.tex, L1479-L1486 (Table I, CIFAR-10, alpha=0.5):
  FedTGP: 71.70 | FedPLVM: 69.76 | FedCOF: 69.64 | FedORGP: 68.70 | FedSA: 68.48 | FedGMKD: 56.37 | FedSCD: 76.66

  % Overleaf/ieee2026.tex, L1581 (Table II, CIFAR-10, alpha=0.5):
  \quad Baseline (CE + Augmentation only)                 & 72.79 & -3.87 \\
  ```
- **Phân tích chi tiết**:
  1. **Bất công bằng nghiêm trọng trong tiền xử lý dữ liệu huấn luyện**: Rà soát toàn bộ thư mục `system/flcore/clients/` và `scripts/prepare_all_datasets.py` chứng minh hàm tăng cường ảnh lật ngang ngẫu nhiên + cắt ngẫu nhiên sau đệm phản xạ 4 pixel (`_apply_scd_view`) **chỉ được cài đặt độc quyền bên trong `clientSCD` (`clientscd.py:70–84, 446`)**. Tất cả 6 thuật toán đối chứng (`FedTGP`, `FedPLVM`, `FedGMKD`, `FedSA`, `FedORGP`, `FedCOF`) đều huấn luyện trên **tensor ảnh tĩnh nguyên bản không hề có augmentation và không xáo trộn (`shuffle=False`)** suốt 100 vòng! Cách viết câu `L1455–L1456` như một thiết lập chung làm người đọc hiểu lầm rằng mọi thuật toán đều dùng chung phép tăng cường dữ liệu này.
  2. **Nghịch lý tự tố cáo ngay giữa `Table II` và `Table I`**: Trên CIFAR-10 ($\alpha = 0.5$), dòng cuối của `Table II` (`Baseline (CE + Augmentation only)`) — thực chất là các client tự huấn luyện cục bộ độc lập **không hề truyền bất kỳ thông tin liên kết nào lên server** — đã đạt tới **`72.79%`**, tức **vượt qua toàn bộ 6 thuật toán Federated Learning trong `Table I`** (cao hơn cả baseline mạnh nhất `FedTGP` `71.70%` tới `+1.09` điểm phần trăm và cao hơn `FedCOF` truyền toàn bộ trọng số mạng `69.64%` tới `+3.15` điểm phần trăm)! Bất kỳ phản biện IEEE nào đặt `Table I` cạnh `Table II` cũng sẽ phát hiện ngay rằng `+1.09` pp trong tổng mức cải thiện `+4.96` pp của `FedSCD` so với `FedTGP` đến từ việc chỉ mình `FedSCD` có data augmentation + head mixup chứ không phải từ chia sẻ thống kê liên kết.
- **Đề xuất cách sửa cụ thể**:
  - **Giải pháp chuẩn mực nhất cho tạp chí IEEE Transactions**: Đưa phép tăng cường ảnh chuẩn (random horizontal flip + random crop với padding 4 pixel) và `shuffle=True` vào pipeline nạp dữ liệu chung `Client.load_train_data` trong `clientbase.py` để cả 6 baseline và `FedSCD` thi đấu trên cùng một mặt bằng dữ liệu đầu vào, rồi cập nhật lại `Table I`.
  - **Trong trường hợp chưa chạy lại ngay bộ benchmark 6 baseline**: Phải ghi rõ tại `L1455–L1456` rằng phép tăng cường ảnh trên GPU và mixup đầu phân loại là một phần trong quy trình huấn luyện cục bộ của `FedSCD` (chưa bật trong code gốc của các baseline), giải thích thẳng thắn sự chênh lệch giữa dòng cơ sở cục bộ `72.79%` ở `Table II` với các baseline không dùng augmentation ở `Table I`, đồng thời bổ sung dòng ablation `FedSCD w/o Input Augmentation (--scd_aug 0)` vào `Table II`.

---

### `EXP-03` — Đặt Tên Sai Bản chất Dòng `"Baseline (CE + Augmentation only)"` trong `Table II` (Ẩn Giao thức Không Giao tiếp FL + Đang Bật Feature Manifold Mixup) và Thiếu Ablation Cho Các Trụ cột Cốt lõi ($B_t^c = 0$, $\gamma_{\max} = 0$, $g^c \equiv \mathbf{1}$)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1570–L1591` (`Table II` & Section V-C) và `L1626–L1640`; đối chiếu `ABLATION_RUN_GUIDE.md` (`L42–L44, L107`) và `system/flcore/clients/clientscd.py` (`L64, L412–L414, L460–L470`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1581 (Table II):
  \quad Baseline (CE + Augmentation only)                 & 72.79 & -3.87 \\
  ```
  Đối chiếu lệnh chạy thực tế trong `ABLATION_RUN_GUIDE.md` (`L107`):
  ```powershell
  @{ Name = "Baseline_CE_Aug"; Extra = @("-lp", "0", "--lamda_virt_global", "0", "--lamda_virt_local", "0") }
  ```
- **Phân tích chi tiết**:
  1. **Nhãn `"Baseline (CE + Augmentation only)"` sai ở hai điểm quan trọng**: Khi chạy `FedSCD` với `-lp 0 --lamda_virt_global 0 --lamda_virt_local 0`: (a) **Feature Manifold Mixup trên đầu phân loại vẫn đang bật hoàn toàn** (`--mixup_alpha = 0.2`, $\rho_{\mathrm{mix}} = 0.25$, `clientscd.py:64, 460–470`) cùng kiến trúc tối ưu tách rời 2 optimizer, nên không hề phải *"CE + Augmentation only"*; và (b) toàn bộ kênh chia sẻ thống kê toàn cục bị ngắt, biến mô hình thành **huấn luyện cục bộ độc lập không có học liên kết (Local-Only, Zero Communication)** nhưng bảng và lời văn (`L1591`) không hề nói đây là mô hình không giao tiếp liên kết.
  2. **Thiếu thực nghiệm bóc tách riêng cho nhiều đóng góp chính được tuyên bố từ Mở đầu**: `Tables II–III` chưa kiểm chứng độc lập: (a) **Số hạng độ phân tán liên client $B_t^c = 0$** (so sánh trung bình phương sai ngây thơ với định lý phương sai toàn phần — vốn là Đóng góp 3 ở `L291–L300`, buộc tác giả phải viết đoạn tự thú nhận ở `L1634–L1638`); (b) **Co cụm mỏ neo Empirical Bayes** ($\gamma_{\max} = 0$, tức dùng thẳng $\mu^c$ không trộn $\mu_k^c$); (c) **Cổng tín hiệu mỏ neo** ($g^c \equiv \mathbf{1}$); (d) **Feature Manifold Mixup** ($\rho_{\mathrm{mix}} = 0$); và (e) tách riêng đóng góp của đặc trưng ảo cục bộ ($\lambda_{\mathrm{vl}} = 0$) so với đặc trưng ảo toàn cục ($\lambda_{\mathrm{vg}} = 0$).
- **Đề xuất cách sửa cụ thể**:
  Đổi tên hàng cuối `Table II` (`L1581`) thành `\quad Local-Only Decoupled Baseline (CE + Aug. + Head Mixup, no FL)` (hoặc chạy lại với `--mixup_alpha 0` và ghi rõ `Local-Only`), đồng thời bổ sung các dòng ablation còn thiếu (`w/o Between-Client Scatter ($B_t^c = 0$)`, `w/o Anchor Shrinkage ($\gamma_{\max}=0$)`, `w/o Signal Gate ($g^c = \mathbf{1}$)`, `w/o Head Mixup ($\rho_{\mathrm{mix}}=0$)`) để lấp đầy khoảng trống bằng chứng thực nghiệm.

---

### `EXP-04` — Toàn bộ Bảng Kết quả Chỉ Chạy Một Seed Duy nhất Không Có Sai số Chuẩn (`mean ± std`) và `Table III` Mang Tên *"Sensitivity to ... hyperparameters"* Nhưng Chỉ Thử 3 Công tắc Bật/Tắt Nhị phân
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1463–L1464`, `L1466–L1490` (`Table I`), `L1562–L1584` (`Table II`), `L1594–L1617` (`Table III`).
- **Trích dẫn nguyên văn**:
  ```latex
  \caption{Sensitivity to design choices and hyperparameters on CIFAR-10 ($\alpha=0.5$). \NMD{This table seems not to be significant too. Actually, if you want to see sensitivity to design choices, you need to vary the hyper parameters and show the convergence curves. Each figure shows convergence curves of FedCSD with different hyper-parameter while other hyper-params fixed. Please also write discussion about it.}}
  ```
- **Phân tích chi tiết**:
  1. **Thiếu độ lệch chuẩn qua nhiều hạt giống ngẫu nhiên (multi-seed uncertainty)**: Mọi con số trong `Tables I, II, III` đều chỉ lấy từ 1 seed duy nhất (`seed=2006`, `-t 1`). Dưới phân hoạch Dirichlet non-IID với $K=20$ client, phương sai giữa các seed thường dao động từ $\pm 0.5\%$ đến $\pm 2.0\%$. Do đó, các khoảng cách nhỏ như `+0.31` pp trên Tiny-ImageNet ($\alpha=0.1$), `+0.37` pp trên CIFAR-100 ($\alpha=1.0$), hay các mức giảm `-0.50` pp ($\beta=1.0$) và `-0.55` pp ($\rho=0$) trong `Table III` chưa đủ ý nghĩa thống kê nếu chỉ chạy 1 lần.
  2. **Tiêu đề `Table III` không khớp nội dung**: `Table III` tuyên bố khảo sát *"Sensitivity to design choices and hyperparameters"*, nhưng thực tế không hề quét dải giá trị (sweep) của bất kỳ siêu tham số liên tục nào ($\lambda, \lambda_{\mathrm{ins}}, \beta, \rho, \lambda_{\mathrm{vl}}, \lambda_{\mathrm{vg}}, \kappa$), mà chỉ bật/tắt đúng 3 lựa chọn nhị phân (`proj_mode=class`, `rho=0`, `beta=1.0`) — đúng như nhận xét xác đáng của đồng tác giả `\NMD` tại `L1595`.
- **Đề xuất cách sửa cụ thể**:
  - Chạy tối thiểu 3 seed độc lập (ví dụ `2006, 2007, 2008` qua cờ `-t 3`) để báo cáo `mean ± std` cho các bảng chính.
  - Xóa bình luận `\NMD{...}` ở `L1595`, đồng thời: (a) bổ sung biểu đồ/bảng quét độ nhạy thực sự theo dải giá trị của $\lambda \in \{0.05, 0.1, 0.3, 0.5\}$, $\beta \in \{0.1, 0.3, 0.6, 1.0\}$, $\lambda_{\mathrm{ins}} \in \{0, 0.25, 0.75, 1.0\}$; hoặc (b) nếu chưa bổ sung sweep, gộp 3 dòng của `Table III` vào `Table II` hoặc đổi tên `Table III` thành `Ablation of Statistical Reference Design Choices on CIFAR-10 ($\alpha=0.5$)` (bỏ cụm từ *"Sensitivity to ... hyperparameters"*).

---

### `EXP-05` — Năm Ô Số liệu Baseline trong `Table I` Lệch Làm tròn / Sao chép (`0.01–0.02` pp) So với Log CSV Thực nghiệm Gốc
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1481–L1483` (`Table I`); đối chiếu `Figure_Plot/wandb_seed2006_max_acc.csv` (`L4, L10, L18, L23, L36`) và `results/runs_results/*.csv`. *(Lưu ý: các lỗi định dạng hiển thị như `21.8` một chữ số thập phân, `{31.00}` thừa ngoặc, `-0.9` và dấu gạch nối ASCII `-` được gộp tại `FIG-05` trong Phần C.)*
- **Trích dẫn nguyên văn**:
  ```latex
  FedGMKD \cite{zhang2024fedgmkd} & 40.00 & 56.37 & 59.53 & ...
  FedSA \cite{zhou2025fedsa}      & 88.79  & 68.48 & 60.89 & 40.87 & ...
  FedORGP \cite{guo2025fedorgp}   & \underline{88.88} & 68.70 & 61.21 & 46.34 & 26.83 & 20.36 & ...
  ```
- **Phân tích chi tiết**:
  Đối chiếu tự động từng ô trong `Table I` với giá trị `max_acc` chính xác trong `results/runs_results/*.csv` và `Figure_Plot/wandb_seed2006_max_acc.csv` phát hiện **5 ô baseline bị lệch `0.01–0.02` điểm phần trăm** do cắt phần thập phân (truncation) thay vì làm tròn chuẩn hoặc gõ nhầm chữ số cuối (thứ hạng in đậm/gạch chân không bị ảnh hưởng):
  1. `FedGMKD` — CIFAR-10 ($\alpha=0.1$): in `40.00` vs. giá trị thực `39.9800%` $\rightarrow$ chuẩn là **`39.98`**.
  2. `FedGMKD` — CIFAR-10 ($\alpha=0.5$): in `56.37` (cắt cụt) vs. giá trị thực `56.3766%` $\rightarrow$ làm tròn chuẩn là **`56.38`**.
  3. `FedORGP` — CIFAR-10 ($\alpha=1.0$): in `61.21` vs. giá trị thực `61.2048%` $\rightarrow$ làm tròn chuẩn là **`61.20`**.
  4. `FedSA` — CIFAR-100 ($\alpha=0.1$): in `40.87` vs. giá trị thực `40.8649%` $\rightarrow$ làm tròn chuẩn là **`40.86`**.
  5. `FedORGP` — CIFAR-100 ($\alpha=1.0$): in `20.36` vs. giá trị thực `20.3372%` $\rightarrow$ làm tròn chuẩn là **`20.34`**.
- **Đề xuất cách sửa cụ thể**:
  Cập nhật chính xác 5 ô trên trong `Overleaf/ieee2026.tex` (`L1481–L1483`) thành `39.98`, `56.38`, `61.20`, `40.86`, và `20.34`.

---

### `EXP-06` — Không Công bố Việc `FedPLVM` và `FedGMKD` Cũng Gom Toàn bộ Tham số Mạng CNN ($\mathcal{O}(|\theta|)$) Mỗi Vòng, Thiếu Siêu tham số Riêng của Cả 6 Baseline ở Mục IV, và Bỏ sót Nhận xét Hành vi của `FedORGP` trên `Figure 1`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L199–L209` (`Fig. 1`), `L716–L728` (Section III-B), `L1449–L1454` (Section IV), `L1541–L1546` (Section V-B); đối chiếu `system/flcore/servers/serverplvm.py` (`L90`), `servergmkd.py` (`L101`), `system/run_target_experiments.py` (`L18–L23`), và `Figure_Plot/prototype_margin.csv` & `bhattacharyya_overlap.csv` (`L102`). *(Xem thêm `FIG-03` ở Phần C về lỗi vẽ thiếu đường cong `FedPLVM` trên `Fig. 1`.)*
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L1449-L1450:
  We compare FedSCD against FedTGP~\citep{zhang2024fedtgp}, FedPLVM~\citep{wang2024fedplvm}, FedGMKD~\citep{zhang2024fedgmkd}, FedSA~\citep{zhou2025fedsa}, FedORGP~\citep{guo2025fedorgp}, and FedCOF~\citep{goswami2025fedcof}. For FedCOF, we adopt an iterative fine-tuning adaptation that couples local cross-entropy training and backbone aggregation with covariance-based classifier calibration across rounds.

  % Overleaf/ieee2026.tex, L207 & L1544:
  FedTGP has lower overlap than FedSCD while also having lower accuracy.
  ```
- **Phân tích chi tiết**:
  1. **Chỉ khai báo gom trọng số mạng cho `FedCOF` mà bỏ quên `FedPLVM` và `FedGMKD`**: Dòng `L1450` nói rõ rằng `FedCOF` có gom tham số backbone qua các vòng, gây ấn tượng rằng 5 baseline còn lại đều là các phương pháp nhẹ không truyền trọng số mạng như `FedSCD`. Tuy nhiên, kiểm tra `serverplvm.py:90` (`self.aggregate_parameters()`) và `servergmkd.py:101` (`self.aggregate_parameters()`) xác nhận **cả `FedPLVM` lẫn `FedGMKD` đều truyền và trung bình hóa toàn bộ trọng số mạng CNN mỗi vòng** (chi phí truyền thông $\mathcal{O}(|\theta|)$, lớn hơn hàng bậc độ lớn so với nhóm chỉ truyền thống kê lớp $\mathcal{O}(Cd)$ gồm `FedSCD`, `FedTGP`, `FedSA`, `FedORGP`)! Việc nêu rõ điều này không chỉ đảm bảo tính chính xác học thuật mà còn làm nổi bật hơn ưu thế hiệu quả truyền thông của `FedSCD` trước `FedCOF`, `FedPLVM` và `FedGMKD`.
  2. **Hoàn toàn thiếu cấu hình siêu tham số đặc thù của 6 baseline trong Section IV**: Bài báo liệt kê chi tiết siêu tham số của `FedSCD` nhưng không nêu bất kỳ siêu tham số nào đã dùng cho 6 baseline (`run_target_experiments.py:18–23`: `FedTGP` dùng `-lam 10.0 -se 100 -mart 100`; `FedSA` `-lam 0.1 -mcl 0.01 -cc 1.0`; `FedPLVM` `--plvm_lamb 100.0 --plvm_tau 0.07`; `FedORGP` `-clam 1.0 -slam 1.0 -gam 1.0 --orgp_server_epochs 1`; `FedGMKD` `--gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6`; `FedCOF` `--cof_gamma 1.0`).
  3. **Mô tả thiếu `FedORGP` khi phân tích `Fig. 1` (`L207, L721, L1544`)**: Tại vòng 100 trong dữ liệu của `Fig. 1` (`prototype_margin.csv` & `bhattacharyya_overlap.csv` `L102`), không chỉ `FedTGP` (`margin=4.66, overlap=0.0075, acc=71.70%`) mà cả **`FedORGP`** (`margin=4.72, overlap=0.0362, acc=68.50%`) cũng đồng thời có biên Euclid lớn hơn và điểm chồng lấn thấp hơn `FedSCD` (`margin=2.62, overlap=0.0620, acc=76.66%`) nhưng độ chính xác kiểm thử thấp hơn nhiều.
- **Đề xuất cách sửa cụ thể**:
  Tại Section IV (`L1450–L1454`), nêu rõ `FedPLVM`, `FedGMKD` và `FedCOF` trao đổi tham số mạng $\mathcal{O}(|\theta|)$ bên cạnh thông tin lớp/phân phối trong khi `FedTGP`, `FedSA`, `FedORGP` và `FedSCD` chỉ trao đổi tóm tắt mức lớp $\mathcal{O}(Cd)$, đồng thời bổ sung ngắn gọn các siêu tham số chuẩn của 6 baseline; tại `L207, L721, L1544`, nhắc tới `FedTGP and FedORGP` thay vì chỉ mình `FedTGP`.

---

### Tiểu mục Đối chiếu Trạng thái Hồi quy so với Báo cáo Kiểm toán ICML Trước đó (`PAPER_REVIEW_REPORT_VI.md`)
Để giúp nhóm tác giả nắm bắt tiến độ khắc phục từ phiên bản `icml2026.tex` sang `ieee2026.tex`:
- **Các lỗi lớn đã được khắc phục triệt để trong `Overleaf/ieee2026.tex`**: Đã bổ sung đầy đủ chứng minh nội dòng cho cả 3 Proposition (`CRIT-02`); đã điền kín toàn bộ 84/84 ô số liệu trong `Table I` (`CRIT-05`); đã sửa toàn bộ phép tính chênh lệch điểm phần trăm trong văn bản và loại bỏ các baseline "ma" FedProto/CCVR không có trong bảng (`CRIT-06`); đã sửa dứt điểm lỗi tham chiếu hỏng `(Eq. ??)` (`CRIT-07`); đã công bố tường minh cơ chế Feature Manifold Mixup trong phương trình đầu phân loại ở Section III-D và Section IV (`MAJ-02`); đã sửa diễn đạt trọng số nghịch đảo phương sai/độ phân tán (`MAJ-03, MAJ-11`), tách ký hiệu phương sai gộp $\bar{v}_t^c$ (`MAJ-04`), sửa phạm vi trung bình toàn cục của $\tau_v$ (`MAJ-07`), thêm bộ lọc $i \in \mathcal{I}$ vào $\mathcal{L}_{\mathrm{PL}}$ (`MAJ-08`), cập nhật độ phức tạp tương phản thành $\mathcal{O}((B+C)^2 d)$ (`MAJ-10`), nêu rõ phép chia $1/n_k^c$ không hiệu chỉnh Bessel trong Proposition~\ref{prop:exact} (`MAJ-12`), và đồng bộ kích thước minibatch $B=32$ ở đợt chạy `seed=2006` (`MAJ-14`).
- **Các lỗi còn tồn đọng hoặc biến tướng sang dạng mới**: Báo cáo `max_acc` không công bố kèm che khuất sự sụp đổ/crash của `FedTGP` và `FedPLVM` vẫn còn nguyên, nay phát sinh thêm lỗi tham chiếu treo sang "Section IV limitations" không tồn tại (`CRIT-04` $\rightarrow$ `EXP-01`); lệch siêu tham số biến tướng theo chiều ngược lại khi bài báo chuyển sang bộ trọng số mới tạo ra `76.66%` nhưng quên cập nhật `system/` và `ABLATION_RUN_GUIDE.md` (`MAJ-01` $\rightarrow$ `CODE-02`) kèm mâu thuẫn mới ở công thức $n_0$ (`CODE-03`); bảng đơn seed và `Table III` công tắc nhị phân vẫn chưa khắc phục (`MAJ-17` $\rightarrow$ `EXP-04`); dòng cơ sở của `Table II` vẫn giữ tên gây hiểu lầm trong khi đang bật head mixup và ngắt giao tiếp FL (`MAJ-18` $\rightarrow$ `EXP-03`); và các macro biên tập `\NMD{...}` cùng chữ đỏ `\textcolor{red}{...}` vẫn tồn tại trên diện rộng (`CRIT-03` $\rightarrow$ `IEEE-01`).

---

## Phần C — Chuẩn Định dạng IEEE (`IEEEtran`), Biên dịch LaTeX, Tài liệu Tham khảo (`ieee2026.bib`), Hình vẽ & Bảng biểu (28 Lỗi: `IEEE-01` – `VER-01`)

> *Ghi chú khử trùng lặp mã định danh*: Mã `FIG-06` (trục hoành 101 bước đánh chỉ số từ `0..100` trên `Fig. 1` và `Fig. 3` so với $t=1,\dots,100$ trong `Algorithm 1`) đã được hợp nhất vào lỗi căn nguyên **`CODE-01`** (Phần B).

### `IEEE-01` — Cờ Ngày Nháp (`-New-12/11`, `New 13/11`), Bình luận Nội bộ Đồng Tác giả (`\NMD{...}`) và ~35 Khối Văn bản Tô Đỏ (`\textcolor{red}{...}`) Hiển thị Trực tiếp trên Tiêu đề, Caption, Bookmark và Thân Bài PDF
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`:
  - Khai báo macro màu & nháp ở preamble: `L54–L60`
  - 7 tiêu đề mục/tiểu mục bị nhiễm nhãn nháp/màu đỏ: `L318`, `L644`, `L754`, `L808`, `L1119`, `L1261`, `L1557`
  - 5 bình luận nội bộ `[NMD: ...]` in xanh đậm trong văn bản & caption `Table III`: `L398`, `L441`, `L496`, `L1532`, `L1595`
  - ~35 đoạn văn bản bọc `\textcolor{red}{...}`: `L582`, `L586–L589`, `L602–L604`, `L633–L636`, `L656–L659`, `L696–L701`, `L709–L714`, `L718–L721`, `L730–L731`, `L745–L748`, `L764–L765`, `L773–L776`, `L783`, `L826–L829`, `L854–L856`, `L875–L876`, `L924–L928`, `L948–L950`, `L977–L978` (lồng kép `\textcolor{red}{\textcolor{red}{...}}`), `L1043–L1050`, `L1105–L1109`, `L1111–L1114`, `L1128–L1129`, `L1166–L1171`, `L1182–L1187`, `L1190`, `L1207–L1211`, `L1213–L1214`, `L1233–L1239`, `L1255–L1256`, `L1258–L1260`, `L1266–L1271`.
- **Trích dẫn nguyên văn**:
  ```latex
  L318:  \section{Related Work-New-12/11}
  L644:  \subsection{Separating Means Is Not Separating Distributions \textcolor{red}{New13/11}}
  L977:  \textcolor{red}{\textcolor{red}{For each positive pair, the denominator retains ...}}
  L1532: ... \NMD{Figure 3 and this paragraph is not significant. The reason is because it gives no insights.}
  L1557: \subsection{\textcolor{red}{Ablation Study}}
  L1595: \caption{Sensitivity to design choices and hyperparameters on CIFAR-10 ($\alpha=0.5$). \NMD{This table seems not to be significant too. ... convergence curves of FedCSD ...}}
  ```
- **Phân tích chi tiết**:
  Bản thảo đang ở trạng thái đang sửa dở (work-in-progress markup): các chuỗi `-New-12/11` và `New 13/11` xuất hiện trực tiếp trong tiêu đề mục in hoa của IEEE (`II. RELATED WORK-NEW-12/11`) và làm hỏng mục lục PDF (PDF bookmarks); 5 ghi chú phê bình nội bộ `[NMD: ...]` in chữ đậm màu xanh dương ngay trong phần Related Work, kết quả thực nghiệm và cả tiêu đề `Table III` (bên trong còn chứa lỗi chính tả gõ nhầm tên phương pháp thành `FedCSD` và lỗi ngữ pháp `Figure 3 and this paragraph is`); cùng khoảng 35 khối chữ màu đỏ rực trải dài suốt 5 trang phương pháp. Bất kỳ tạp chí IEEE nào cũng sẽ **từ chối sơ loại ngay lập tức (desk-reject)** khi nhận bản nộp còn nguyên dấu vết bình luận nội bộ và chữ đánh dấu nháp.
- **Đề xuất cách sửa cụ thể**:
  Xóa toàn bộ hậu tố `-New-12/11` và `\textcolor{red}{New 13/11}` khỏi các lệnh `\section`/`\subsection` (`\section{Related Work}`, `\subsection{Separating Means Is Not Separating Distributions}`, `\subsection{Overview of FedSCD}`, `\subsection{Client Side: Distribution-Aware Prototype Learning}`, `\subsection{Server Side: Statistics-Consistent Aggregation}`, `\subsection{Algorithm and Complexity}`, `\subsection{Ablation Study}`); xử lý nội dung và xóa sạch cả 5 lệnh `\NMD{...}`; gỡ bỏ toàn bộ lớp bọc `\textcolor{red}{...}` về văn bản đen chuẩn; và xóa khối macro nháp `L54–L60` ở preamble.

---

### `IEEE-02` — Giữ Nguyên Văn Thông tin Tác giả, Cơ quan và Tên Tạp chí Giả định của Template (`Firstname1 Lastname`, `Department of XXX, University of YYY`, `IEEE Transactions on ...`)
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L67–L73` (và khối tiểu sử mẫu đang comment ở `L1661–L1663`).
- **Trích dẫn nguyên văn**:
  ```latex
  \author{Firstname1~Lastname%
  \thanks{Manuscript created 2026. Firstname1 Lastname is with the Department of XXX, University of YYY, Location, Country (e-mail: first1.last1@xxx.edu). Corresponding author: Firstname1 Lastname.}%
  }

  \markboth{IEEE Transactions on ...,~Vol.~XX, No.~X, 2026}%
  {Lastname: Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning}
  ```
- **Phân tích chi tiết**:
  Khác với hội nghị ICML dùng chế độ ẩn danh hai chiều, tạp chí IEEE Transactions áp dụng phản biện một chiều (single-blind) hoặc yêu cầu khai báo đầy đủ danh tính tác giả theo quy chuẩn `New_IEEEtran_how-to.tex` (`L100–L122`). Việc để nguyên văn thông tin giả `Firstname1 Lastname`, `Department of XXX, University of YYY`, `first1.last1@xxx.edu` và tiêu đề chạy đầu trang `IEEE Transactions on ...` là lỗi thủ tục nghiêm trọng dẫn tới **desk-reject tự động** tại cổng ScholarOne / IEEE Author Portal.
- **Đề xuất cách sửa cụ thể**:
  Điền đầy đủ danh sách tác giả thực tế kèm hạng thành viên `\IEEEmembership{...}` (nếu có), thông tin đơn vị công tác, địa chỉ và email trong `\thanks{...}`, tên tạp chí IEEE mục tiêu ở đối số thứ nhất của `\markboth`, và mẫu `FirstAuthorLastName \MakeLowercase{\textit{et al.}}: Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning` ở đối số thứ hai của `\markboth` khi có từ 2 tác giả trở lên:
  ```latex
  \author{Author1~Name,~\IEEEmembership{Member,~IEEE,} Author2~Name, and Author3~Name%
  \thanks{Manuscript received Month DD, 2026. The authors are with the Wireless Communications Laboratory (WicomLab), Department of ..., University of ..., City, Country (e-mail: \{author1, author2, author3\}@domain.edu). \textit{(Corresponding author: AuthorX Name.)}}%
  }

  \markboth{IEEE Transactions on Neural Networks and Learning Systems,~Vol.~XX, No.~X, Month~2026}%
  {Name \MakeLowercase{\textit{et al.}}: Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning}
  ```

---

### `IEEE-03` — Lệnh `\paragraph{Title.}` Tàn dư Từ ICML Dưới Lớp `IEEEtran` Sinh Dấu Chấm Kép `.:`, Nhảy Cấp Phân mục (`III-D0a`), và Trộn Lẫn Kiểu Tiêu đề Chạy Ngang (`\textbf{Heading.}`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L838`, `L930`, `L980`, `L1022` (Section III-D); `L1134`, `L1189`, `L1218` (Section III-E); so với `\textbf{Heading.}` tại `L1264`, `L1336`, `L1436`, `L1443`, `L1446`, `L1449`, `L1452`, `L1457`, `L1499`, `L1513`, `L1523`, `L1541`, `L1548`, `L1586`, `L1619`.
- **Trích dẫn nguyên văn**:
  ```latex
  L838:  \paragraph{Client-adapted anchors and dispersion reweighting.}
  L930:  \paragraph{Dispersion-aware angular learning.}
  L980:  \paragraph{Coordinate-wise Mahalanobis compactness.}
  L1022: \paragraph{Decoupled optimization and classifier calibration.}
  L1134: \paragraph{Moment pooling with between-client scatter.}
  L1189: \paragraph{Temporal smoothing of statistical references.}
  L1218: \paragraph{Dispersion- and support-aware temperature calibration.}
  ```
- **Phân tích chi tiết**:
  Trong `IEEEtran.cls`, hệ thống phân cấp tiêu đề gồm `\section` (`I.`), `\subsection` (`A.`), `\subsubsection` (`1)` — cấp thứ 3), và `\paragraph` (`a)` — cấp thứ 4), trong đó **cả `\subsubsection` lẫn `\paragraph` đều tự động thêm dấu hai chấm `:` vào cuối tiêu đề**. Do giữ nguyên thói quen đặt dấu chấm `.` cuối `\paragraph{... .}` của template ICML:
  1. Cả 7 tiêu đề ở trang 6–8 của PDF bị in ra **dấu câu kép xấu xí `.:`** (ví dụ: `a) Client-adapted anchors and dispersion reweighting.:`).
  2. Vì bỏ qua cấp `\subsubsection`, bộ đếm phân mục trong `ieee2026.aux` và PDF bookmark bị nhảy cóc dị dạng thành `III-D0a`, `III-E0a`.
  3. Không đồng nhất cấu trúc toàn bài: Section III-D/E dùng `\paragraph{...}`, trong khi Section III-F/G, IV và V lại dùng `\textbf{Heading.}` thủ công không đánh số.
- **Đề xuất cách sửa cụ thể**:
  Đổi toàn bộ 7 lệnh `\paragraph{Heading.}` (bỏ dấu chấm cuối!) thành `\subsubsection{Heading}` (hiển thị chuẩn `1) Client-adapted anchors and dispersion reweighting:`), đồng thời chuyển các đầu mục con cùng cấp trong Section III-F, IV và V sang `\subsubsection{Heading}` để đồng bộ chuẩn IEEE Transactions.

---

### `IEEE-04` — Khai báo Bộ Đếm Định lý Theo `[section]` Dưới Chữ Số La Mã của `IEEEtran` Tạo Nhãn Phi Chuẩn `Proposition III.1`–`III.3` (Cùng 5 Môi trường Định lý Khai báo Thừa)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L42–L51` (ảnh hưởng tới các mệnh đề tại `L1373, L1390, L1410` và các tham chiếu tại `L701, L1170, L1171, L1437`).
- **Trích dẫn nguyên văn**:
  ```latex
  \theoremstyle{plain}
  \newtheorem{theorem}{Theorem}[section]
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{corollary}[theorem]{Corollary}
  \theoremstyle{definition}
  \newtheorem{definition}[theorem]{Definition}
  \newtheorem{assumption}[theorem]{Assumption}
  \theoremstyle{remark}
  \newtheorem{remark}[theorem]{Remark}
  ```
- **Phân tích chi tiết**:
  Vì `IEEEtran.cls` định nghĩa `\thesection` bằng chữ số La Mã (`I`, `II`, `III`), tùy chọn `[section]` ở `L43` khiến LaTeX sinh ra nhãn lai La Mã – Ả Rập `Proposition III.1`, `Proposition III.2`, `Proposition III.3`. Theo quy chuẩn biên tập của IEEE (`New_IEEEtran_how-to.tex` `L474–L481`), các mệnh đề, bổ đề và định lý trong bài báo IEEE Transactions được đánh số Ả Rập liên tục xuyên suốt toàn bài (`Proposition 1`, `Proposition 2`, `Proposition 3`). Ngoài ra, 6 môi trường `theorem`, `lemma`, `corollary`, `definition`, `assumption`, `remark` được khai báo nhưng không hề sử dụng lần nào trong bài.
- **Đề xuất cách sửa cụ thể**:
  Thay thế khối `L42–L51` bằng khai báo gọn chuẩn IEEE:
  ```latex
  \theoremstyle{plain}
  \newtheorem{proposition}{Proposition}
  ```

---

### `IEEE-05` — Nạp Các Gói Can thiệp Bố cục Trái Khuyến nghị của `IEEEtran.cls` (`todonotes` + `\marginparwidth`, `enumitem`, `float`, Tắt `balance`) VÀ Để Dòng Trống Bên Trong `itemize` (`L265–L314`) Phá Vỡ Danh sách Gốc của `IEEEtran`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L17` (`%\usepackage{balance}`), `L20–L28` (`xcolor[table]`, `comment`, `float`, `dirtytalk`, `enumitem`, `xspace`), `L58–L59` (`\setlength{\marginparwidth}{2cm}`, `todonotes`), và `L265–L314` (các dòng trống `L266, L278, L290, L301, L313` bên trong `\begin{itemize}...\end{itemize}`); đối chiếu `Overleaf/IEEEtran.cls` (`L2262–L2263`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.tex, L17, L23-L28, L58-L59:
  %\usepackage{balance}
  ...
  \usepackage{float}
  \usepackage{dirtytalk}
  \usepackage{bm}
  \usepackage{enumitem}
  \usepackage{cite}
  \usepackage{xspace}
  ...
  \setlength{\marginparwidth}{2cm}
  \usepackage[textsize=tiny]{todonotes}

  % Overleaf/IEEEtran.cls, L2262-L2263:
  % DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
  % AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
  ```
- **Phân tích chi tiết**:
  1. **`\setlength{\marginparwidth}{2cm}` + `todonotes` (`L58–L59`)**: Thay đổi thông số lề trang của `IEEEtran` (2 cm = 0.79 in vượt quá lề ngoài 0.625 in của IEEE) và nạp toàn bộ hệ thống TikZ/PGF nặng nề dù cả bài không có bất kỳ lệnh `\todo` nào.
  2. **`enumitem` (`L26`) ghi đè cơ chế danh sách chuẩn của `IEEEtran` + Bẫy dòng trống tại `L265–L314`**: Gói `enumitem` ghi đè các thanh ghi thụt lề chuyên dụng của IEEE (`\IEEElabelindent`, `\IEEEilabelindent`) dù bài báo không dùng bất kỳ tùy chọn nào của `enumitem`. **Đặc biệt lưu ý khi gỡ `enumitem`**: trong khối liệt kê đóng góp `\begin{itemize}` (`L265–L314`) hiện có **5 dòng trống** tại `L266, L278, L290, L301` và nhất là `L313` ngay trước `\end{itemize}`. Mã nguồn `Overleaf/IEEEtran.cls:2262–2263` cảnh báo rõ rằng môi trường danh sách gốc `\@IEEEitemize` của IEEE **cấm để dòng trống bên trong** (nếu để dòng trống, đặc biệt trước `\end{itemize}`, `IEEEtran` sẽ ép ngắt đoạn mới làm lệch khoảng cách dọc và thụt đầu dòng sau danh sách)!
  3. **Gói thừa & chưa cân bằng cột trang cuối**: `float` (`L23`) không tương thích tốt với cơ chế float của IEEE (`New_IEEEtran_how-to.tex` khuyến nghị dùng `stfloats` nếu cần bảng đôi ở đáy trang); `comment` (`L21`), `dirtytalk` (`L24`), `xspace` (`L28`) và tùy chọn `[table]` của `xcolor` (`L20`) hoàn toàn không dùng đến; trong khi `\usepackage{balance}` (`L17`) lại bị comment tắt đi, khiến hai cột tài liệu tham khảo ở trang cuối (p. 13) bị lệch cao thấp.
- **Đề xuất cách sửa cụ thể**:
  Gỡ bỏ `float`, `dirtytalk`, `enumitem`, `comment`, `xspace`, `\setlength{\marginparwidth}{2cm}`, `todonotes` và tùy chọn `[table]` trên `xcolor`; mở comment `\usepackage{balance}` (đặt `\balance` ở cột trái trang cuối trước mục tài liệu tham khảo cuối); **đồng thời xóa sạch 5 dòng trống `L266, L278, L290, L301, L313` bên trong khối `\begin{itemize}...\end{itemize}`**.

---

### `IEEE-06` — Tùy chọn Lớp Tài liệu `lettersize` Không Tồn tại trong `IEEEtran.cls` v1.8b Gây Cảnh báo `Unused global option(s): [lettersize]`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1` (ghi nhận tại `Overleaf/ieee2026.log`, `L909–L910`).
- **Trích dẫn nguyên văn**:
  ```latex
  \documentclass[lettersize,journal]{IEEEtran}
  ```
- **Phân tích chi tiết**:
  File `Overleaf/IEEEtran.cls` (phiên bản 1.8b, dòng `L297`) chỉ định nghĩa tùy chọn `\DeclareOption{letterpaper}` (vốn đã là mặc định trong chế độ `journal`), không có tùy chọn nào tên là `lettersize` (đây là lỗi đánh máy nổi tiếng trong một bản template mẫu năm 2020 của IEEE). Do đó mỗi lần biên dịch, LaTeX đều phát sinh cảnh báo `LaTeX Warning: Unused global option(s): [lettersize].`
- **Đề xuất cách sửa cụ thể**:
  Sửa dòng `L1` thành `\documentclass[letterpaper,journal]{IEEEtran}` hoặc `\documentclass[journal]{IEEEtran}`.

---

### `IEEE-07` — Tham số Định vị Float Dùng `[t]` Thay vì `[!t]` Theo Chuẩn IEEE (Làm Trễ Bảng/Hình & Sinh `Underfull \vbox`), Không Đồng nhất `Figure` vs. `Fig.`, và Ký hiệu Toán trong Abstract
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L77` (Abstract), `L194, L199` (`fig:motivation`), `L716, L759, L789` (`fig:framework`), `L1273` (`alg:fedscd`), `L1516, L1524, L1527, L1544` (`fig:fedscd_4datasets`), `L1562` (`tab:ablation_components`), `L1593` (`tab:ablation_params`).
- **Trích dẫn nguyên văn**:
  ```latex
  L199:  \begin{figure}[t]
  L789:  \begin{figure*}[t]
  L1273: \begin{algorithm}[t]
  L1516: \begin{figure}[t]
  L1562: \begin{table}[t]
  L1593: \begin{table}[t]
  ```
- **Phân tích chi tiết**:
  1. Chỉ duy nhất `Table I` (`L1465`) dùng `[!t]`, còn cả 6 khối hình, bảng và thuật toán còn lại chỉ viết `[t]`. Tài liệu `New_IEEEtran_how-to.tex` (`L181, L188`) quy định rõ luôn dùng `[!t]` để LaTeX bỏ qua giới hạn tỷ lệ diện tích float mặc định và đặt hình/bảng sát đỉnh cột chuẩn IEEE, hạn chế kéo giãn khoảng trắng dọc `Underfull \vbox`.
  2. Tham chiếu hình không tuân thủ quy chuẩn viết tắt của IEEE: `L759` và `L1544` viết `Fig.~\ref{...}`, trong khi `L194, L716, L1524, L1527` viết đầy đủ `Figure~\ref{...}`. Theo *IEEE Editorial Style Manual* chính thức, tạp chí IEEE sử dụng dạng viết tắt **`Fig.~1` ngay cả khi đứng ở đầu câu** (khác với `Equation` khi mở đầu câu). Ngoài ra, `New_IEEEtran_how-to.tex` (`L134`) khuyến nghị hạn chế tối đa ký hiệu LaTeX phức tạp (`$\mathcal{O}(Cd)$`) trong Abstract để tránh lỗi hiển thị metadata trên IEEE Xplore.
- **Đề xuất cách sửa cụ thể**:
  Đổi `[t]` thành `[!t]` ở cả 6 môi trường float (`L199, L789, L1273, L1516, L1562, L1593`), và đổi toàn bộ `Figure~\ref{...}` tại `L194, L716, L1524, L1527` thành `Fig.~\ref{...}` theo đúng chuẩn biên tập IEEE Transactions.

---

### `LATEX-01` — Nạp File Macro Thừa Từ ICLR `Overleaf/math_commands.tex` Ghi đè Phá Hỏng Lệnh `\eqref` của `amsmath`, Định nghĩa Lỗi `\rvu`/`\rvi`, Để Sót Placeholder `\method` và Gọi Lại `\usepackage`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L39` (`\input{math_commands.tex}`); và `Overleaf/math_commands.tex`, `L3`, `L20–L31`, `L35–L70`, `L124`, `L136`.
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/math_commands.tex:
  L3:   \usepackage{amsmath,amsfonts,bm}
  L20:  \providecommand{\method}{\textsc{[Method]}}
  L49:  \def\eqref#1{equation~\ref{#1}}
  L124: \def\rvu{{\mathbf{i}}}
  L136: \def\rvu{{\mathbf{u}}}
  ```
- **Phân tích chi tiết**:
  `Overleaf/math_commands.tex` là file macro dài 522 dòng còn sót lại từ template ICLR và **không có bất kỳ macro nào trong đó được sử dụng trong `ieee2026.tex`**. Nguy hiểm hơn:
  1. Vì `math_commands.tex` được `\input` tại dòng `L39` *sau khi* đã nạp `\usepackage{amsmath}`, dòng `L49` (`\def\eqref#1{equation~\ref{#1}}`) âm thầm **phá hủy hoàn toàn lệnh `\eqref{...}` chuẩn của `amsmath`** (biến `(1)` thành `equation 1` trái quy chuẩn IEEE). Hiện tại `ieee2026.tex` gõ thủ công `(\ref{...})` nên lỗi này đang tiềm ẩn (latent landmine), nhưng bất kỳ chỉnh sửa nào dùng `\eqref{...}` (theo khuyến nghị của `New_IEEEtran_how-to.tex` `L787`) sẽ lập tức làm hỏng định dạng tham chiếu phương trình.
  2. File này gọi lại `\usepackage{amsmath,amsfonts,bm}` bên trong file con (`L3`), gõ nhầm `\def\rvu{{\mathbf{i}}}` ở `L124` rồi ghi đè ở `L136` (làm mất `\rvi`), chứa placeholder `\textsc{[Method]}` (`L20`), và định nghĩa hàng loạt macro vector in đậm (`\muv, \zv, \xv`, `L23–L30`) mâu thuẫn với quy ước vector không in đậm của bài báo.
- **Đề xuất cách sửa cụ thể**:
  Xóa dòng `\input{math_commands.tex}` tại `L39` của `Overleaf/ieee2026.tex` (và sau đó có thể chuyển toàn bộ `(\ref{eq:...})` sang `\eqref{eq:...}` chuẩn của `amsmath`).

---

### `LATEX-02` — Macro Giả lập `natbib` (`\citep` / `\citet`) Ánh xạ `\citet` Thành `[N]` Trần Gây Sai Ngữ pháp Nếu Dùng, Kèm Khoảng trắng Có Thể Ngắt Dòng Trước `\cite` trong `Table I`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L32–L36` (và các lệnh `\citep` rải rác từ `L89` đến `L1450` so với `\cite` trong `Table I` `L1479–L1484`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Compatibility macros for natbib commands in IEEEtran
  \makeatletter
  \newcommand{\citep}[2][]{\if\relax\detokenize{#1}\relax\cite{#2}\else\cite[#1]{#2}\fi}
  \newcommand{\citet}[2][]{\if\relax\detokenize{#1}\relax\cite{#2}\else\cite[#1]{#2}\fi}
  \makeatother
  ```
- **Phân tích chi tiết**:
  Đây là đoạn vá tạm khi chuyển bài từ ICML (`natbib`) sang `IEEEtran` (`cite`). Việc định nghĩa `\citet` giống hệt `\cite` khiến mọi lệnh trích dẫn chủ ngữ `\citet{key}` chỉ in ra số `[5]` thay vì `Zhang et al. [5]`, tạo câu sai ngữ pháp kiểu `"[5] propose..."`. Đồng thời, trong thân bài dùng `~\citep{...}` nhưng trong `Table I` (`L1479–L1484`) lại viết `FedTGP \cite{zhang2024fedtgp}` với dấu cách thường không có dấu ngã `~` chống ngắt dòng.
- **Đề xuất cách sửa cụ thể**:
  Thay toàn bộ `\citep{...}` trong `Overleaf/ieee2026.tex` bằng lệnh chuẩn IEEE `~\cite{...}`, thêm `~` trước `\cite` ở cả 6 dòng của `Table I` (`FedTGP~\cite{zhang2024fedtgp}`), và xóa bỏ đoạn shim `L32–L36`.

---

### `LATEX-03` — Thiếu `\label` cho `\section{Experimental Setup}` (`L1441`), 7 Tham chiếu Cứng `Section IV` (6 Vị trí Thiếu `~`) và 3 Cách Trỏ Mơ hồ Bằng Lời (`"in the experimental details"`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1441` (thiếu `\label`), 7 tham chiếu cứng tại `L207`, `L598`, `L1437`, `L1500`, `L1514`, `L1532`, `L1542`, và 3 tham chiếu bằng lời không số mục tại `L727–L728`, `L835–L836`, `L1088`.
- **Trích dẫn nguyên văn**:
  ```latex
  L1441: \section{Experimental Setup}
  L207:  ... defined in Section IV, and panel (c) ...
  L598:  ... exclusion specified in Section IV.
  L727:  ... in the experimental metric definitions.
  L835:  ... specified in the experimental details.
  L1088: ... as specified in the implementation details.
  ```
- **Phân tích chi tiết**:
  Tất cả các mục cấp cao nhất của bài báo đều có `\label`, ngoại trừ duy nhất Section IV (`L1441`). Hệ quả là 7 vị trí phải gõ cứng chuỗi chữ `Section IV` không có siêu liên kết `\ref` (vi phạm quy tắc bắt buộc dùng tham chiếu động của `New_IEEEtran_how-to.tex` `L787–L791`, trong đó 6/7 chỗ còn quên dấu `~` nên dễ bị rớt chữ `IV` xuống đầu dòng mới), và 3 vị trí khác trong Section III (`L727–L728, L835–L836, L1088`) phải né tránh bằng cụm từ mơ hồ *"in the experimental details"* / *"in the implementation details"* thay vì trỏ chính xác tới mục thiết lập thực nghiệm.
- **Đề xuất cách sửa cụ thể**:
  Thêm `\label{sec:exp_setup}` ngay dưới `\section{Experimental Setup}` tại `L1441`, và thay toàn bộ 10 vị trí nêu trên bằng tham chiếu chuẩn `Section~\ref{sec:exp_setup}`.

---

### `LATEX-04` — Đặt Nhầm `\vspace{-10pt}` Ngay Sau Môi trường Nổi `\end{table*}` (`L1491`) Làm Sập Khoảng Trắng Phía Trên Tiêu đề Section V ở Trang 10, Cùng `\vspace{1.5mm}` Thủ công trong `Tables II & III`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1490–L1492`, `L1566`, `L1597`.
- **Trích dẫn nguyên văn**:
  ```latex
  L1490: \end{table*}
  L1491: \vspace{-10pt}
  L1492: \section{Experimental Evaluations}
  ...
  L1564: \caption{Component ablation on CIFAR-10 ($\alpha=0.5$).}
  L1565: \label{tab:ablation_components}
  L1566: \vspace{1.5mm}
  ```
- **Phân tích chi tiết**:
  1. Vì `table*` (`Table I`) là một khối float trôi lên đầu trang 11, lệnh `\vspace{-10pt}` đặt bên ngoài `\end{table*}` tại `L1491` **hoàn toàn không thu hẹp khoảng cách dưới Bảng I**, mà thực thi trực tiếp vào luồng văn bản cột của trang 10 ngay trước `\section{Experimental Evaluations}` — làm tiêu đề `V. EXPERIMENTAL EVALUATIONS` bị dính sát lên đoạn văn phía trên một cách biến dạng!
  2. Tương tự, lệnh `\vspace{1.5mm}` đặt thủ công dưới `\caption` của `Table II` (`L1566`) và `Table III` (`L1597`) ghi đè lên khoảng cách chuẩn `\@IEEEtablecaptionsepspace` của `IEEEtran.cls` (trong khi `Table I` không có lệnh này, tạo khoảng cách giữa caption và đường kẻ bảng không đồng nhất giữa các bảng).
- **Đề xuất cách sửa cụ thể**:
  Xóa hoàn toàn dòng `\vspace{-10pt}` tại `L1491` và xóa hai dòng `\vspace{1.5mm}` tại `L1566` và `L1597`.

---

### `LATEX-05` — Đánh số Thừa 5 Phương trình Phụ Không Nhãn `(6)–(10)` Trong Khi Bỏ Trống Số Cho Các Phương trình Cốt lõi `\[...\]` (Làm Lệch Số Phương trình và Sai Tham chiếu `(17)` ở Caption `Fig. 2`), Nhãn `\label` Thừa, và Dấu Hai Chấm Trước Phương trình Hiển thị
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L616, L685, L797, L866–L906, L937, L990, L1001–L1007, L1066–L1071, L1076–L1085, L1175–L1181, L1192, L1223, L1245, L1301–L1302, L1418, L1459–L1463, L1493, L1497, L1538, L1558, L1651`.
- **Trích dẫn nguyên văn**:
  ```latex
  % 5 dòng phương trình trung gian được đánh số (6)-(10) nhưng không có \label:
  L866-L874: \begin{align} v^c&=... \\ \tau_v&=... \end{align}
  L880-L890: \begin{align} s^c &= ... \\ \widetilde{g}^c &= ... \end{align}
  L898-L906: \begin{align} \bar{u}= ... \end{align}

  % Công thức cốt lõi lại không đánh số \[ ... \], khiến caption Fig. 2 (L797) trỏ sai sang (17):
  L797: ... virtual features generated as $\tilde{z}^{c,b}=\Pi(p^c+\eta_b\epsilon\odot\tilde{\sigma}^c)$ with $\epsilon\sim\mathcal{N}(0,\mathbf I)$, following (\ref{eq:head_loss}).
  ```
- **Phân tích chi tiết**:
  1. **Mất cân đối chính sách đánh số phương trình & lệch tài liệu hướng dẫn**: Ba khối `align` tại `L866–L906` sinh ra 5 số phương trình `(6)` đến `(10)` không hề có `\label` và không bao giờ được gọi tới (khiến toàn bộ phương trình phía sau bị đẩy lệch `+5` đơn vị so với `Overleaf/FedSCD_Figure_Guide.md` và `FedSCD_Master_Flow_Blueprint.md`, đồng thời làm tham chiếu gộp `(\ref{eq:anchor})--(\ref{eq:proj})` tức `(5)–(11)` ở `Algorithm 1` `L1302` ngầm nhảy qua định nghĩa nội dòng của $u^c$ ở `L895` và $m$ ở `L908`). Ngược lại, các công thức trung tâm của thuật toán — đặc biệt là công thức sinh đặc trưng ảo $\widetilde{z}^{c,b}$ (`L1077–L1085`), độ lệch chuẩn chính quy hóa server $\bar{\sigma}_t^c$ (`L1175–L1181`), phân rã phương sai–độ chệch độ nén (`L1001–L1007`), $\mathcal{L}_{\mathrm{data}}$ (`L1066–L1071`) và chỉ số đánh giá chính $A_{\mathrm{local}}$ (`L1459–L1463`) — lại bị bỏ không đánh số trong `\[...\]`. Hệ quả trực tiếp là caption `Fig. 2` (`L797`) phải gõ lại công thức $\tilde{z}^{c,b}$ rồi viết sai là *"following (17)"* (trong khi `(17)` là hàm mất mát $\mathcal{L}_{\mathrm{head}}$, không phải công thức sinh $\tilde{z}^{c,b}$!), và `Algorithm 1` bước 19 (`L1323`) cũng phải nhồi công thức $\bar{\sigma}_t^c$ vào dòng hẹp gây tràn/giãn dòng.
  2. **Vi phạm quy tắc dấu câu trước phương trình của IEEE**: Tại 7 vị trí (`L685`, `L898`, `L937`, `L1076`, `L1192`, `L1223`, `L1245`), câu dẫn kết thúc bằng dấu hai chấm `:` ngay trước phương trình hiển thị. Theo *IEEE Editorial Style Manual*, phương trình hiển thị phải đóng vai trò là thành phần ngữ pháp liền mạch của câu văn dẫn nhập mà không dùng dấu hai chấm sau mệnh đề hoàn chỉnh.
  3. **Nhãn `\label` không được tham chiếu**: `eq:mean` (`L616`), `eq:dist` (`L990`), `eq:bayes_bound` (`L1418`) và 5 nhãn mục ở cuối bài chưa được gọi (có thể giữ nhãn mục để mở rộng nhưng cần rà soát nhãn phương trình).
- **Đề xuất cách sửa cụ thể**:
  Chuyển các bước trung gian không trích dẫn ở `L866–L906` sang `align*` (hoặc gắn `\nonumber`), đánh số chính thức kèm nhãn `\label{eq:virt_feat}`, `\label{eq:server_std}`, `\label{eq:metric_acc}` cho các phương trình cốt lõi ở `L1077–L1085`, `L1175–L1181`, `L1459–L1463`, sửa tham chiếu ở `L797` thành `(\ref{eq:virt_feat})`, và bỏ dấu hai chấm `:` trước cả 7 phương trình hiển thị (ví dụ tại `L936–L937`: `The ProtoNCE objective relates each valid sample to all available class anchors via` rồi xuống phương trình).

---

### `LATEX-06` — Xuống Dòng Trước `~\citep{...}` Tạo Khoảng Trắng Kép (`L111–L112`, `L153–L154`), Ngoặc Nhọn Bóp Khoảng Cách Toán tử `{=}`/`{+}`/`{-}` và Lẫn Lộn Chỉ số Dưới `\text` vs. `\mathrm`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L111–L112`, `L153–L154`, `L977–L978`, `L1444–L1456`, `L1535`, `L1576–L1579`. *(Lưu ý: lỗi dính chữ sau dấu phẩy `fluctuate,altering` tại `L1190` được tách riêng tại `PROSE-02` trong Phần D.)*
- **Trích dẫn nguyên văn**:
  ```latex
  L111-L112: investigated under label skew and class imbalance
             ~\citep{dai2023fednh,zhang2024fedtgp},
  L153-L154: feature augmentation, and representation learning
             ~\citep{luo2021ccvr,tang2024fedimpro,
  L1444:     Data are partitioned among $K{=}20$ clients ...
  L1535:     a complete uplink contains $C(2d{+}1)$ scalar values ...
  L1576:     \quad w/o Compactness ($\mathcal{L}_{\text{PL}}$) ...
  ```
- **Phân tích chi tiết**:
  1. **Khoảng trắng kép trước trích dẫn (`L111–L112`, `L153–L154`)**: Đặt dấu `~` ở đầu dòng mới ngay sau ký tự xuống dòng không có `%` làm TeX chèn **2 khoảng trắng liên tiếp** (1 dấu cách từ ký tự xuống dòng sau từ `imbalance` / `learning` cộng thêm 1 khoảng trắng cứng `~`), đồng thời cho phép ngắt dòng xấu ngay trước `~[6]`. (Tại `L977–L978` cũng có hai khoảng trắng kép trong mã nguồn.)
  2. **Bóp nghẹt khoảng cách toán học bằng `{=}`, `{+}`, `{-}` (`L1444–L1456` xuất hiện 24 lần, và `L1535`)**: Bọc dấu bằng/cộng/trừ trong ngoặc nhọn (`$K{=}20$`, `$C(2d{+}1)$`, `$\xi{=}\max(u, 1{-}u)$`) triệt tiêu khoảng đệm chuẩn `\thickmuskip`/`\medmuskip` của TeX, mâu thuẫn ngay với `$K=20$` (`L1466`), `$\alpha=0.5$` (`L77`) và `$C(2d+1)$` (`L1337`).
  3. **Trộn lẫn `\text{...}` và `\mathrm{...}` cho tên viết tắt chỉ số dưới**: Toàn bộ Section III viết `\mathcal{L}_{\mathrm{PL}}`, `\mathcal{L}_{\mathrm{NCE}}`, `\lambda_{\mathrm{ins}}`, `T_{\mathrm{warm}}`, nhưng Section IV (`L1453`) và `Table II` (`L1576–L1579`) lại viết `\lambda_{\text{ins}}`, `T_{\text{warm}}`, `\mathcal{L}_{\text{PL}}` (lệnh `\text{...}` sẽ bị in nghiêng theo ngữ cảnh nếu đặt trong tiêu đề/mệnh đề in nghiêng, khác với `\mathrm{...}` luôn đứng thẳng).
- **Đề xuất cách sửa cụ thể**:
  Nối liền `~\cite{...}` sát vào từ đứng trước (`imbalance~\cite{...}`, `learning~\cite{...}`), gỡ bỏ các cặp ngoặc `{=}`, `{+}`, `{-}` trong văn bản toán nội dòng (`$K=20$`, `$C(2d+1)$`), và chuẩn hóa toàn bộ chỉ số tên toán tử/hàm mất mát về `\mathrm{...}`.

---

### `LATEX-07` — File `Overleaf/ieee2026.pdf`/`.log` Lưu Sẵn Đã Lỗi thời & Chưa Hội tụ Tham chiếu Chéo, Lỗi `Underfull \hbox (badness 10000)` ở Bước 19 `Algorithm 1`, Thiếu Font `pcrr7t` (`missfont.log`) và Các File Rác Xung đột Đồng bộ `.aux`/`.out`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.log` (`L1040, L1049, L1069, L1079`), `Overleaf/missfont.log` (`L1`), `Overleaf/ieee2026 2.aux`, `Overleaf/ieee2026 2.out`, `Overleaf/ieee2026 3.aux`, và `Overleaf/ieee2026.tex` (`L1320–L1325`).
- **Trích dẫn nguyên văn**:
  ```text
  % Trong file log cũ đang lưu sẵn Overleaf/ieee2026.log (biên dịch ngày 17/11):
  L1040: Overfull \hbox (4.69038pt too wide) detected at line 877
  L1049: Underfull \hbox (badness 10000) in paragraph at lines 1312--1317
  L1069: Underfull \vbox (badness 7433) has occurred while \output is active []
  L1079: LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.

  % Trong lần biên dịch mới nhất từ Overleaf/ieee2026.tex hiện tại (ngày 18/11):
  Underfull \hbox (badness 10000) in paragraph at lines 1320--1325

  % Overleaf/missfont.log (L1):
  mktextfm pcrr7t
  ```
- **Phân tích chi tiết**:
  1. **Bản PDF và file `.log` đang lưu trong `Overleaf/` đã cũ so với `.tex` và dừng trước lượt chạy hội tụ thứ 4**: `Overleaf/ieee2026.pdf` và `ieee2026.log` (ngày 17/11) vẫn chứa hình t-SNE cũ ở trang 11, lỗi tràn dòng `Overfull \hbox (4.69pt)` tại dòng `L877` cũ (`ieee2026.log:1040`), `Underfull \hbox` tại dòng `L1312–L1317` cũ (`ieee2026.log:1049`), `Underfull \vbox` (`ieee2026.log:1069`), và cảnh báo chưa hội tụ nhãn `Label(s) may have changed` (`ieee2026.log:1079`).
  2. **Lỗi `Underfull \hbox (badness 10000)` vẫn tồn tại khi biên dịch mới `Overleaf/ieee2026.tex` hiện tại**: Trên bản `.tex` hiện hành, chính bước 19 của `Algorithm 1` (`ieee2026.tex:1320–1325`: `Pool $(\bar{\mu}_t^c,\bar{v}_t^c)$ using (18)--(19); set $\bar{\sigma}_t^c=[\max(\bar{v}_t^c,\epsilon_v\mathbf{1})]^{\odot1/2}$.`) chứa biểu thức toán nội dòng dài không có điểm ngắt ở cuối cột hẹp, gây giãn rỗng khoảng cách từ với mức phạt tối đa `badness 10000` trên trang 9.
  3. **Thiếu font Courier (`pcrr7t`) cục bộ và file rác Google Drive**: `Overleaf/missfont.log` ghi lại lỗi `mktextfm pcrr7t` do môi trường BasicTeX cục bộ thiếu gói font Adobe Courier mà `IEEEtran.cls` khai báo làm `\ttdefault` (sẽ gây lỗi biên dịch ngay nếu gọi `\url{...}` hoặc `\texttt{...}`). Ngoài ra thư mục `Overleaf/` đang chứa 3 file xung đột đồng bộ Google Drive (`ieee2026 2.aux`, `ieee2026 2.out`, `ieee2026 3.aux`).
- **Đề xuất cách sửa cụ thể**:
  Tách câu lệnh bước 19 của `Algorithm 1` (`L1320–L1325`) thành 2 bước ngắn hoặc trỏ tới phương trình `(\ref{eq:server_std})` như đề xuất tại `ALGO-04`; xóa `missfont.log`, `ieee2026 2.aux`, `ieee2026 2.out`, `ieee2026 3.aux`; cài gói TeX `courier` trên máy cục bộ (nếu biên dịch offline); và chạy đủ chu trình `latexmk -pdf` để triệt tiêu hoàn toàn cảnh báo `Label(s) may have changed`.

---

### `FIG-01` — Năm Sai sót Toán học, Nhân quả Chỉ số Vòng và Luồng Kiến trúc Ngay Bên Trong Sơ đồ Tổng quan `Overleaf/figures/framework.pdf` (`Figure 2`) So với Phương trình Mục III
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/figures/framework.pdf` và `Overleaf/ieee2026.tex`, `L789–L800` (`Figure 2`); đối chiếu `L538–L539, L768–L770, L1028, L1077–L1085, L1111–L1114, L1193–L1203`.
- **Trích dẫn nguyên văn**:
  - Các nhãn hiển thị bên trong `Overleaf/figures/framework.pdf`:
    - Ô *Classifier* (góc dưới bên trái): `$\mathcal{N}(p^c, \tilde{\sigma}^c)$`
    - Ô *EMA Temporal Smoothing* (góc trên bên phải): `$(1-\beta)_{t-1} + (\beta)_t \longrightarrow (\mu_t^c, \sigma_t^c, n_t^c)$`
    - Ô *Weighted Hypersphere* (giữa phía trên): `$\mathcal{L}_{\mathrm{NCE}}(\phi_t^c) + \lambda_{\mathrm{ins}}\mathcal{L}_{\mathrm{SC}}(\tau)$` và nhãn góc `$\phi_t^c$`; Chú giải: `Global moments $(\mu_t^c, \sigma_t^c)$`
    - Ô *Unnormalized space*: `$\mathcal{L}_{\mathrm{PL}}(z, p^c; g^c, v^c) + \ell_{\mathrm{CE}}(h_k(z), y)$` (nhưng chỉ vẽ duy nhất đường truyền $r = \mathrm{sg}[z] \to h_k$)
  - Caption tại `Overleaf/ieee2026.tex`, `L797`: `\caption{Overview of \textsc{FedSCD}. ...}`
- **Phân tích chi tiết**:
  Kiểm tra trực quan từng hộp chức năng và công thức vẽ trong `Overleaf/figures/framework.pdf` phát hiện **5 sai lệch nghiêm trọng** đối với nội dung toán học của bài báo:
  1. **Nhầm độ lệch chuẩn với ma trận hiệp phương sai trong ký hiệu phân phối Gauss**: Hộp *Classifier* viết `$\mathcal{N}(p^c, \tilde{\sigma}^c)$`, truyền thẳng vector độ lệch chuẩn $\tilde{\sigma}^c$ vào vị trí tham số phương sai/hiệp phương sai, trái ngược hoàn toàn với quy ước $\mathcal{N}(\mu^c, \mathrm{diag}((\sigma^c)^{\odot 2}))$ ở `L538–L539` và `L663–L664`. Mũi tên sinh $\tilde{z}^{c,b}$ cũng bỏ sót cả hệ số biên độ nhiễu $\eta_b$ lẫn toán tử cắt không âm $\Pi(\cdot)$ ở `L797, L1078`.
  2. **Biểu thức EMA sai cú pháp toán học**: Hộp *EMA Temporal Smoothing* viết `$(1-\beta)_{t-1} + (\beta)_t$`, gắn chỉ số thời gian $t-1$ và $t$ trực tiếp vào hai hằng số vô hướng $(1-\beta)$ và $\beta$ thay vì gắn vào các biến trạng thái, đồng thời bỏ sót phép làm tròn nguyên $\operatorname{round}(\cdot)$ trên $n_t^c$ của `Eq. (20)`.
  3. **Vi phạm tính nhân quả chỉ số vòng ($\phi_t^c$ thay vì $\phi_{t-1}^c$) và khuyết nhãn $\mu^c$**: Ở vòng $t$, khối huấn luyện cục bộ của client (*Weighted Hypersphere*) lại hiển thị nhiệt độ $\phi_t^c$ — đại lượng chỉ được server tính *sau khi* kết thúc huấn luyện cục bộ, tải lên và gom thống kê của vòng $t$ (`Algorithm 1` bước 21)! Đúng như `L769` đã viết, tại vòng $t$ client chỉ có thể dùng thống kê của vòng $t-1$ là $(\mu_{t-1}^c, \sigma_{t-1}^c, \phi_{t-1}^c)$ (hoặc viết lược bỏ chỉ số vòng là $\phi^c$ như `Eq. (12)`). Ngoài ra, trong hộp *Fixed Empirical-Bayes References*, biểu tượng trung bình toàn cục phía trên bị khuyết nhãn chữ $\mu^c$ (trong khi biểu tượng $\sigma^c$ bên dưới có nhãn).
  4. **Thiếu đường lan truyền thuận không chặn gradient từ $z$ qua $h_k$**: Khối *Unnormalized space* viết số hạng $\ell_{\mathrm{CE}}(h_k(z), y)$ trong $\mathcal{L}_{\mathrm{enc}}$, nhưng trên sơ đồ đầu phân loại $h_k$ chỉ nhận đầu vào duy nhất đã bị chặn gradient $r = \mathrm{sg}[z]$, hoàn toàn không có mũi tên nào nối $z$ (không có `sg`) đi qua $h_k$ để truyền gradient phân loại về cập nhật $f_k$ (`L1113–L1114`).
  5. **Kiểu chữ `\textsc{FedSCD}` ở caption (`L797`)**: Hiển thị thành chữ in hoa nhỏ `FEDSCD`, lệch với cách viết `FedSCD` ở mọi vị trí khác.
- **Đề xuất cách sửa cụ thể**:
  Vẽ và xuất lại file vector `Overleaf/figures/framework.pdf`: (i) sửa nhãn Gauss ở hộp Classifier thành $\mathcal{N}\big(p^c, \mathrm{diag}((\eta_b\widetilde{\sigma}^c)^{\odot 2})\big)$ kèm phép cắt $\Pi(\cdot)$; (ii) sửa hộp EMA thành $(1-\beta)s_{t-1}^c + \beta\bar{s}_t^c$ (với $\operatorname{round}(\cdot)$ cho $n_t^c$); (iii) đổi $\phi_t^c$ ở phía client thành $\phi^c$ (hoặc $\phi_{t-1}^c$) và bổ sung nhãn $\mu^c$ còn thiếu; (iv) bổ sung một nhánh nét đứt từ $z$ đi qua $h_k$ (đóng băng $\theta_{h_k}$) tới $\ell_{\mathrm{CE}}(h_k(z), y)$; và (v) sửa `\textsc{FedSCD}` tại `L797` thành `FedSCD`.

---

### `FIG-02` — Lệch Hoàn toàn Cấu trúc & Tên File Giữa `Overleaf/figures/framework.pdf` Thực tế và Hai Tài liệu Đặc tả `Overleaf/FedSCD_Figure_Guide.md` / `FedSCD_Master_Flow_Blueprint.md`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/figures/framework.pdf` đối chiếu với `Overleaf/FedSCD_Figure_Guide.md` (`L1–L125`) và `Overleaf/FedSCD_Master_Flow_Blueprint.md` (`L1–L81`).
- **Trích dẫn nguyên văn**:
  ```markdown
  # Overleaf/FedSCD_Figure_Guide.md & FedSCD_Master_Flow_Blueprint.md:
  - Tham chiếu các file: figures/framework.drawio, figures/framework_style4_fedplvm.pdf, figures/framework_style5_master.pdf
  - Quy định bố cục 3 cột tuần tự (a)–(g) / Stage 1–3 với nhãn Space 1 / Space 2 và số phương trình cũ Eq. 6, Eq. 11, Eq. 12...
  ```
- **Phân tích chi tiết**:
  Cả hai file hướng dẫn nằm ngay trong thư mục `Overleaf/` mô tả một sơ đồ 3 cột nối tiếp từ trái sang phải có đánh nhãn bước `(a)–(g)`, `Space 1` / `Space 2` và trỏ tới các file `.drawio` / `framework_style4_fedplvm.pdf` / `framework_style5_master.pdf` không hề tồn tại trong `Overleaf/figures/`. Trong khi đó, file `Overleaf/figures/framework.pdf` đang nhúng vào bài lại là bố cục 2 khối khác hẳn và cũng lệch số thứ tự phương trình. Khi tải gói mã nguồn lên hệ thống nộp bài của IEEE (IEEE ScholarOne yêu cầu nộp thư mục source sạch), các file `.md` nội bộ lệch phiên bản này không nên nằm lẫn trong thư mục `Overleaf/`.
- **Đề xuất cách sửa cụ thể**:
  Chuyển các file hướng dẫn nháp `FedSCD_Figure_Guide.md` và `FedSCD_Master_Flow_Blueprint.md` ra khỏi thư mục nộp bài `Overleaf/` (hoặc cập nhật đồng bộ theo phiên bản `framework.pdf` cuối cùng sau khi sửa `FIG-01`).

---

### `FIG-03` — `Figure 1` (`Overleaf/figures/motivation.pdf`) Âm thầm Bỏ sót Baseline `FedPLVM`, Thiếu Nhãn Trục Tung ở Cả 3 Panel (Tiêu đề Panel (b) Không Khớp Định nghĩa Rescaled Overlap) và Co Nhỏ Canvas 2 Cột Xuống 50% Làm Chữ Quá Bé (`~6.5pt`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/figures/motivation.pdf`, `Overleaf/ieee2026.tex` (`L199–L209`), và script sinh hình `Figure_Plot/plot_combined_3panels.py` (`L22, L90–L147, L164`).
- **Trích dẫn nguyên văn**:
  ```latex
  \begin{figure}[t]
    \centering
    \IfFileExists{figures/motivation.pdf}{%
      \includegraphics[width=\linewidth]{figures/motivation.pdf}%
    }{...}
    \caption{Anchor separation, Gaussian overlap, and local test accuracy on CIFAR-10 under Dirichlet label skew with $\alpha=0.5$. Panel (a) reports nearest-other-anchor Euclidean margins, panel (b) reports the dimension-rescaled overlap score defined in Section IV, and panel (c) reports test-set-size-weighted local accuracy. ...}
  \label{fig:motivation}
  \end{figure}
  ```
- **Phân tích chi tiết**:
  1. **Âm thầm loại bỏ `FedPLVM` mà không giải thích**: Section IV (`L1450`) và `Table I` đánh giá **6** thuật toán đối chứng, và ngay trong `plot_combined_3panels.py` (`L22`) cũng tải dữ liệu run `o0vj3q9d` của `FedPLVM`, nhưng danh sách vẽ `algos` (`L90–L147`) lại chỉ vẽ 5 baseline + `FedSCD (Ours)` (bỏ `FedPLVM` do run này bị crash ở bước 98) mà không có bất kỳ chú thích nào trong caption `Fig. 1` hay văn bản.
  2. **Thiếu hoàn toàn nhãn trục tung (Y-axis label) ở cả 3 panel & tên tiêu đề Panel (b) gây hiểu nhầm**: Cả 3 biểu đồ con `(a)`, `(b)`, `(c)` đều có nhãn trục hoành `Communication Rounds` nhưng để trống trục tung (khác với `Fig. 3` có ghi `Test Accuracy (%)`). Đồng thời tiêu đề trên đỉnh Panel (b) ghi `Bhattacharyya Overlap`, dễ làm độc giả nhầm với hệ số Bhattacharyya chuẩn $\mathrm{BC} = \exp(-D_B)$ ở `Eq. (4)`, trong khi thực tế dữ liệu vẽ là chỉ số đã co giãn theo căn số chiều $\exp(-D_B^{\mathrm{eval}}/\sqrt{d})$.
  3. **Co nhỏ hình 2 cột xuống 1 cột làm cỡ chữ dưới ngưỡng đọc chuẩn IEEE**: `plot_combined_3panels.py` (`L164`) khởi tạo hình ở khổ rộng toàn trang 2 cột `figsize=(7.16, 6.0)` inch (`505.48pt`), nhưng trong `ieee2026.tex` lại ép vào môi trường 1 cột `figure` rộng `3.5in` (`252pt`, tỷ lệ thu phóng `0.498x`). Việc thu nhỏ 50% làm cỡ chữ số trục `13pt` bị co xuống còn **`~6.5pt`** (nhỏ hơn ngưỡng tối thiểu `8pt` của IEEE Graphics Guidelines) và nét vẽ mảnh chỉ còn `0.65pt`.
- **Đề xuất cách sửa cụ thể**:
  Chỉnh sửa `Figure_Plot/plot_combined_3panels.py` để: (i) vẽ thêm `FedPLVM` (đánh dấu điểm dừng crash tại vòng 98) hoặc ghi rõ lý do lược bỏ `FedPLVM` trong caption `L207`; (ii) thêm nhãn trục tung rõ ràng (`Nearest-Class Margin`, `Rescaled Overlap`, `Local Test Accuracy (%)`) và đổi tiêu đề phụ Panel (b) thành `Rescaled Gaussian Overlap`; và (iii) xuất lại ở đúng kích thước 1 cột `figsize=(3.5, 2.7)` với font `8–9pt` sắc nét (hoặc chuyển sang `\begin{figure*}[!t]` nếu giữ bố cục 3 cột ngang lớn).

---

### `FIG-04` — File Hình Mồ côi Không Được Tham chiếu `Overleaf/figures/ieee_tsne_cifar10_dir05.pdf` (Cũng Thiếu `FedPLVM` và Thiếu Chú giải Màu Lớp / Dấu Sao)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/figures/ieee_tsne_cifar10_dir05.pdf` (vẫn xuất hiện ở trang 11 của bản PDF cũ `Overleaf/ieee2026.pdf`, nhưng đã bị gỡ khỏi `Overleaf/ieee2026.tex`).
- **Trích dẫn nguyên văn**:
  ```text
  Overleaf/ieee2026.log, L1062-L1067 (bản cũ): <figures/ieee_tsne_cifar10_dir05.pdf, id=491, 527.64125pt x 288.42757pt>
  Overleaf/ieee2026.tex hiện tại: 0 lần gọi \includegraphics{...ieee_tsne_cifar10_dir05...}
  ```
- **Phân tích chi tiết**:
  File `Overleaf/figures/ieee_tsne_cifar10_dir05.pdf` hiện nằm thừa trong thư mục `Overleaf/figures/` mà không được nhúng ở bất kỳ đâu trong `ieee2026.tex`. Kiểm tra trực quan file PDF này cũng cho thấy hình chỉ có 6 ô `(a) FedTGP` đến `(f) FedSCD (Ours)` (tiếp tục bỏ sót `FedPLVM`) và không có thanh chú giải (legend/colorbar) giải thích ý nghĩa 10 màu điểm dữ liệu cùng các dấu sao màu đại diện cho mỏ neo lớp.
- **Đề xuất cách sửa cụ thể**:
  Xóa file thừa `Overleaf/figures/ieee_tsne_cifar10_dir05.pdf` khỏi gói nộp bài (hoặc nếu muốn đưa hình trực quan hóa t-SNE quay lại Section V-B `\ref{sec:geometry}`, cần bổ sung ô của `FedPLVM` và chú giải ký hiệu điểm/dấu sao trước khi nhúng).

---

### `FIG-05` — Lỗi Trình bày & Kiểu chữ Bảng Biểu trên `Tables I–III` (Co Giãn Font Bằng `\resizebox`, Lệch Chữ Số Thập phân `21.8` & `-0.9`, Ngoặc Thừa `{31.00}`, Dấu Gạch Ngang Văn bản `-` Thay Vì Dấu Trừ Toán `$-$`, và Không Thống nhất Viết Hoa Caption)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1465–L1490` (`Table I`), `L1562–L1584` (`Table II`), `L1593–L1617` (`Table III`).
- **Trích dẫn nguyên văn**:
  ```latex
  L1466: \caption{Quantitative Label Skew Results in Terms of Test Accuracy (\%) ... Best in Bold and Second with Underline. }
  L1472: \resizebox{\textwidth}{!}{%
  L1481: FedGMKD \cite{zhang2024fedgmkd} & 40.00 & ... & {31.00} & 19.46 & 21.50 & 21.8 \\
  L1564: \caption{Component ablation on CIFAR-10 ($\alpha=0.5$).}
  L1576: \quad w/o Compactness ($\mathcal{L}_{\text{PL}}$)       & 75.01 & -1.65 \\
  L1577: \quad w/o SupCon ($\lambda_{\text{ins}}=0$)             & 75.76 & -0.9 \\
  ```
- **Phân tích chi tiết**:
  1. **Lạm dụng `\resizebox{\textwidth}{!}{...}` ở `Table I` (`L1472`)**: Co giãn hình học toàn bộ bảng làm méo cỡ chữ và độ dày đường kẻ `booktabs` so với chuẩn bảng chữ gốc của IEEE, trong khi dùng `\footnotesize` kết hợp `\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}` vừa khít chiều ngang `\textwidth` mà không cần biến dạng font.
  2. **Không đồng nhất số chữ số thập phân & ngoặc nhọn thừa**: Ô Tiny-ImageNet $\alpha=1.0$ của `FedGMKD` ở `Table I` (`L1481`) in `21.8` (1 chữ số thập phân trong khi cả 83 ô còn lại đều có 2 chữ số thập phân; đồng thời ô `{31.00}` thừa cặp ngoặc nhọn); hàng `w/o SupCon` ở `Table II` (`L1577`) in `-0.9` (1 chữ số thập phân, mâu thuẫn với chính lời văn ở `L1589` viết `0.90` và toàn bộ các dòng $\Delta$ khác).
  3. **Dùng dấu gạch nối ASCII `-` thay vì dấu trừ toán học `$-$`**: Toàn bộ 8 giá trị âm của cột $\Delta$ trong `Table II` (`L1576–L1581`) và `Table III` (`L1608, L1611, L1614`) đều gõ ở chế độ văn bản (`-1.65`), hiển thị thành gạch nối ngắn thay vì dấu trừ toán học chuẩn (`$-1.65$`).
  4. **Không thống nhất quy tắc viết hoa tiêu đề bảng (Caption Case)**: `Table I` (`L1466`) viết hoa chữ cái đầu mỗi từ (Title Case — hiển thị chữ hoa nhỏ Small Caps trong `IEEEtran`, kèm câu cụt chưa tự nhiên `"Best in Bold and Second with Underline. "`), trong khi `Table II` (`L1564`) và `Table III` (`L1595`) lại viết kiểu câu thường (Sentence case).
- **Đề xuất cách sửa cụ thể**:
  Thay `\resizebox` ở `Table I` bằng `\footnotesize` + `tabular*`; sửa `21.8` $\rightarrow$ `21.80` và `{31.00}` $\rightarrow$ `31.00` ở `L1481`; sửa `-0.9` $\rightarrow$ `$-0.90$` ở `L1577` và đặt tất cả số âm cột $\Delta$ trong chế độ toán `$...$`; thống nhất cả 3 caption bảng theo chuẩn Title Case của `IEEEtran`:
  ```latex
  % Table I (L1466):
  \caption{Weighted Local Test Accuracy (\%) on CIFAR-10, CIFAR-100, Flowers102, and Tiny-ImageNet Under Dirichlet Label Skew $\mathrm{Dir}(\alpha)$ ($\alpha \in \{0.1, 0.5, 1.0\}$, $K=20$ Clients). Best Results Are in Bold and Second-Best Are Underlined.}
  % Table II (L1564):
  \caption{Component Ablation Study on CIFAR-10 ($\alpha=0.5$).}
  % Table III (L1595):
  \caption{Ablation of Statistical Reference Design Choices on CIFAR-10 ($\alpha=0.5$).}
  ```

---

### `BIB-01` — Tài liệu Tham khảo Đang Trích dẫn `chen2025hproto` (`[10]`, `ieee2026.bib:400–406`) Là Bài báo Không Tồn tại / Bịa đặt (Cùng Với Entry Chưa Trích dẫn `liu2025fedpeft`, `L393–L398`)
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L400–L406` (được trích dẫn tại `Overleaf/ieee2026.tex`, `L115` thành tài liệu `[10]`) và `Overleaf/ieee2026.bib`, `L393–L398` (chưa trích dẫn).
- **Trích dẫn nguyên văn**:
  ```bibtex
  % Overleaf/ieee2026.bib, L393-L406:
  @inproceedings{liu2025fedpeft,
    title={Federated Parameter-Efficient Fine-Tuning for Pre-trained Backbones under Heterogeneous Edge Clients},
    author={Liu, Ziyue and Zhang, Jing and Yang, Shiqi and Gao, Dashan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
  }

  @article{chen2025hproto,
    title={Federated Prototype Learning for Heterogeneous Model Architectures in Edge Computing},
    author={Chen, Xiaomin and Wang, Hao and Shen, Xuemin},
    journal={IEEE Transactions on Mobile Computing},
    year={2025},
    publisher={IEEE}
  }
  ```
- **Phân tích chi tiết**:
  Tra cứu đối chiếu trên IEEE Xplore, DBLP, Crossref và Google Scholar xác nhận **hoàn toàn không tồn tại bất kỳ bài báo nào có tiêu đề *"Federated Prototype Learning for Heterogeneous Model Architectures in Edge Computing"* của nhóm tác giả *Xiaomin Chen, Hao Wang, Xuemin Shen* trên tạp chí *IEEE Transactions on Mobile Computing*** (bản thân entry này cũng thiếu hoàn toàn `volume`, `number`, `pages`, `doi`). Tương tự, entry `liu2025fedpeft` (`L393–L398`) cũng không hề tồn tại trong kỷ yếu CVPR 2025. Việc trích dẫn một bài báo giả mạo mang tên chính tạp chí thuộc hệ thống IEEE (*IEEE TMC*) trong một bản thảo nộp cho IEEE là lỗi nghiêm trọng bậc nhất về độ tin cậy học thuật, có thể dẫn tới việc bác bỏ bài ngay lập tức khi phản biện hoặc biên tập viên kiểm tra danh mục tài liệu.
- **Đề xuất cách sửa cụ thể**:
  Xóa ngay cả hai entry `chen2025hproto` và `liu2025fedpeft` khỏi `Overleaf/ieee2026.bib`; tại dòng `L114–L115` của `Overleaf/ieee2026.tex` (`and model heterogeneity~\cite{yi2023fedgh,chen2025hproto}`), thay `chen2025hproto` bằng tài liệu có thật đã có sẵn trong bài như `tan2022fedproto` hoặc bổ sung một bài báo *IEEE TMC* 2025 có thật đã xác minh (ví dụ: L. Fu *et al.*, ``Federated domain-independent prototype learning with alignments of representation and parameter spaces for feature shift,'' *IEEE Trans. Mobile Comput.*, vol. 24, no. 9, pp. 9004–9019, Sep. 2025).

---

### `BIB-02` — Tài liệu Tham khảo Đang Trích dẫn `oh2022fedbabu` (`[21]`, `ieee2026.bib:370–375`) Bị Sai Tiêu đề Gốc và Bịa Đặt Bốn Đồng Tác giả Không Có Thật
- **Mức độ nghiêm trọng**: `Critical`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L370–L375` (trích dẫn tại `Overleaf/ieee2026.tex`, `L331`; hiển thị tại `Overleaf/ieee2026.bbl`, `L83–L85`).
- **Trích dẫn nguyên văn**:
  ```bibtex
  @inproceedings{oh2022fedbabu,
    title={{FedBABU}: Towards Enhanced Representation for Federated Learning via Neutralizing the Classifier},
    author={Oh, Jaehoon and Han, Sungwon and Yun, Seok-Ju and Cho, Seojin and Kim, Sunghun},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2022}
  }
  ```
- **Phân tích chi tiết**:
  Bài báo *FedBABU* công bố chính thức tại ICLR 2022 có tiêu đề thật là **`{FedBABU}: Towards Enhanced Representation for Federated Image Classification`** (không phải *"for Federated Learning via Neutralizing the Classifier"*) và danh sách tác giả thật chỉ gồm **3 người là `Jaehoon Oh, Sangmook Kim, Se-Young Yun`** (hoàn toàn không phải 4 cái tên bịa đặt `Sungwon Han, Seok-Ju Yun, Seojin Cho, Sunghun Kim`).
- **Đề xuất cách sửa cụ thể**:
  Thay thế `L370–L375` trong `Overleaf/ieee2026.bib` bằng thông tin chính xác từ kỷ yếu ICLR 2022:
  ```bibtex
  @inproceedings{oh2022fedbabu,
    title     = {{FedBABU}: Towards Enhanced Representation for Federated Image Classification},
    author    = {Oh, Jaehoon and Kim, Sangmook and Yun, Se-Young},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2022}
  }
  ```

---

### `BIB-03` — Mảnh BibTeX Xóa Dở Thiếu Dòng Mở Đầu `@inproceedings{...` trong `Overleaf/ieee2026.bib` (`L380–L385`, Cùng Dòng `year` Bị Ngắt Đôi ở `L322–L323`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L380–L385` và `L322–L323`.
- **Trích dẫn nguyên văn**:
  ```bibtex
    title={{FedGH}: Heterogeneous Federated Learning with Generalized Global Header},
    author={Yi, Liping and Wang, Gang and Liu, Xiaoguang and Shi, Zhuanqiang and Yu, Han},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia (MM)},
    pages={8686--8696},
    year={2023}
  }
  ```
- **Phân tích chi tiết**:
  Khối dòng `L380–L385` bị mất hoàn toàn dòng khai báo đầu mục `@inproceedings{key,` (đây là bản trùng lặp xóa dở của entry `yi2023fedgh` đã có ở `L103–L109`, thậm chí còn ghi sai tên tác giả thứ tư thành `Shi, Zhuanqiang` thay vì tên đúng `Shi, Zhuan` ở `L105`) và kết thúc bằng một dấu ngoặc nhọn đóng mồ côi `}`. Dù trình biên dịch `bibtex` cổ điển bỏ qua văn bản nằm ngoài `@...{...}`, các bộ kiểm tra cú pháp `.bib` nghiêm ngặt (Overleaf BibTeX linter, Biber, JabRef) đều báo lỗi cú pháp ngoặc không cân xứng. Ngoài ra ở `L322–L323`, trường `year` bị ngắt xuống dòng bất thường.
- **Đề xuất cách sửa cụ thể**:
  Xóa hẳn khối thừa `L380–L385` trong `Overleaf/ieee2026.bib` và gộp `L322–L323` về một dòng `year = {2021}`.

---

### `BIB-04` — Thiếu Ngoặc Nhọn `{...}` Bảo vệ Chữ Hoa trong Tiêu đề BibTeX Khiến `IEEEtran.bst` Đổi Thường Sai Các Tên Riêng `GFPL` (`Gfpl`), `FedUP` (`Fedup`), `FedCDWA` (`Fedcdwa`) và `Wasserstein` (`wasserstein`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L408` (`lu2026gfpl`, `[15]`), `L415` (`qi2026fedup`, `[16]`), `L428–L430` (`liu2026fedcdwa`, `[17]`); hiển thị lỗi trực tiếp tại `Overleaf/ieee2026.bbl` (`L65–L73`, trang 13 của PDF).
- **Trích dẫn nguyên văn**:
  ```latex
  % Kết quả biên dịch thực tế trong Overleaf/ieee2026.bbl (L66, L69, L72):
  ``Gfpl: Generative federated prototype learning for resource-constrained and data-imbalanced vision task,''
  ``Fedup: Uncertainty-aware personalized federated learning via probabilistic prototypes,''
  ``Fedcdwa: Decoupled federated prototype distillation with hierarchical wasserstein aggregation,''
  ```
- **Phân tích chi tiết**:
  Kiểu tài liệu tham khảo `IEEEtran.bst` tự động chuyển toàn bộ tiêu đề bài báo về dạng chữ thường (sentence case), ngoại trừ chữ cái đầu tiên của câu và các cụm nằm trong cặp ngoặc nhọn `{...}`. Do các entry mới thêm ở cuối file `.bib` không đặt ngoặc nhọn quanh tên thuật toán và tên nhà toán học, danh mục tài liệu tham khảo ở trang 13 bị in sai thành `Gfpl`, `Fedup`, `Fedcdwa` và `wasserstein`.
- **Đề xuất cách sửa cụ thể**:
  Bọc ngoặc nhọn bảo vệ chữ hoa trong `Overleaf/ieee2026.bib`:
  ```bibtex
  % L408 (lu2026gfpl):
  title = {{GFPL}: Generative Federated Prototype Learning for Resource-Constrained and Data-Imbalanced Vision Task},
  % L415 (qi2026fedup):
  title = {{FedUP}: Uncertainty-Aware Personalized Federated Learning via Probabilistic Prototypes},
  % L428-L430 (liu2026fedcdwa):
  title = {{FedCDWA}: Decoupled Federated Prototype Distillation with Hierarchical {Wasserstein} Aggregation},
  ```

---

### `BIB-05` — Cắt Cụt Danh sách Tác giả Bằng `and others` (`et al.`) Sai Quy định IEEE Đối với Bài 5 Tác giả `lu2026gfpl` (`[15]`) và Thiếu Số Kỳ Kép / Tác giả của `kairouz2021advances` (`[3]`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L23–L31` (`kairouz2021advances`, `[3]`) và `L407–L412` (`lu2026gfpl`, `[15]`).
- **Trích dẫn nguyên văn**:
  ```bibtex
  % L25, L28 (kairouz2021advances):
  author = {Kairouz, Peter and others},
  ...
  number = {1},

  % L409 (lu2026gfpl):
  author = {Lu, Shiwei and others},
  ```
- **Phân tích chi tiết**:
  1. Theo quy chuẩn tài liệu tham khảo của IEEE (`IEEEtran_bst_HOWTO.pdf`), khi bài báo có **từ 6 tác giả trở xuống**, bắt buộc phải liệt kê **đầy đủ tên tất cả các tác giả** (`IEEEtran.bst` tự động xử lý `et al.` khi danh sách vượt ngưỡng). Bài báo `arXiv:2602.21873` (`lu2026gfpl`) chỉ có đúng **5 tác giả** (`Shiwei Lu, Yuhang He, Jiashuo Li, Qiang Wang, Yihong Gong`), nên việc viết `Lu, Shiwei and others` làm `[15]` in sai chuẩn thành `S. Lu et al.`
  2. Bài khảo sát nền tảng `kairouz2021advances` (`[3]`, *Foundations and Trends in Machine Learning*, vol. 14, pp. 1–210, 2021) xuất bản dưới dạng số kỳ kép **`number = {1--2}`** (không phải `number = {1}`), và đối với bài đông tác giả (>6 người), chuẩn IEEE liệt kê đủ 6 tác giả đầu tiên trước `and others` (hoặc để `IEEEtran.bst` tự rút gọn từ danh sách đầy đủ) chứ không cắt ngay sau tác giả thứ nhất một cách tùy tiện khi không nhất quán.
- **Đề xuất cách sửa cụ thể**:
  - Tại `lu2026gfpl` (`L409`): sửa thành `author = {Lu, Shiwei and He, Yuhang and Li, Jiashuo and Wang, Qiang and Gong, Yihong},`.
  - Tại `kairouz2021advances` (`L25, L28`): sửa `number = {1--2},` và điền tối thiểu 6 tác giả đầu `author = {Kairouz, Peter and McMahan, H. Brendan and Avent, Brendan and Bellet, Aur{\'e}lien and Bennis, Mehdi and Bhagoji, Arjun Nitin and others},`.

---

### `BIB-06` — Thiếu Trường `pages` và `volume` ở Năm Bài báo Kỷ yếu Chính thức Đang Được Trích dẫn (`wang2024fedplvm`, `goswami2025fedcof`, `luo2021ccvr`, `collins2021exploiting`, `mcmahan2017fedavg`)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L7–L13` (`mcmahan2017fedavg`, `[1]`), `L144–L150` (`wang2024fedplvm`, `[8]`), `L238–L244` (`luo2021ccvr`, `[13]`), `L256–L261` (`goswami2025fedcof`, `[31]`), `L340–L346` (`collins2021exploiting`, `[19]`).
- **Trích dẫn nguyên văn**:
  ```bibtex
  @inproceedings{wang2024fedplvm, ... booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {37}, year = {2024} }
  @inproceedings{goswami2025fedcof, ... booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2025} }
  ```
- **Phân tích chi tiết**:
  Một số bài hội nghị được trích dẫn đã ghi đầy đủ `volume` và `pages` (như `zhang2024fedgmkd` `[29]` và `mclaughlin2024pfedfda` `[25]` ở NeurIPS vol. 37), nhưng các bài cùng kỷ yếu ngay sát cạnh lại bỏ trống thông tin trang/tập chính thức:
  - `wang2024fedplvm` (`L144–L150`, `[8]`, NeurIPS 2024): thiếu `pages = {88348--88372}`.
  - `goswami2025fedcof` (`L256–L261`, `[31]`, NeurIPS 2025): thiếu `volume = {38}` và `pages = {72500--72534}`.
  - `luo2021ccvr` (`L238–L244`, `[13]`, NeurIPS 2021): thiếu `volume = {34}`.
  - `collins2021exploiting` (`L340–L346`, `[19]`, ICML 2021) & `mcmahan2017fedavg` (`L7–L13`, `[1]`, AISTATS 2017): thiếu số tập PMLR `volume = {139}` và `volume = {54}`.
- **Đề xuất cách sửa cụ thể**:
  Bổ sung đầy đủ các trường `volume` và `pages` đã xác minh ở trên vào 5 mục tương ứng trong `Overleaf/ieee2026.bib`.

---

### `BIB-07` — Entry BibTeX Bị Trùng lặp (`fedup2026` vs. `qi2026fedup`), 17 Entry Không Dùng Đến, Bản `arXiv` Cũ Đã Xuất bản Chính thức (`qi2023fedcspc`, `agcl2023fedgcd`) và Kiểu Định dạng `@article` Cho `arXiv preprint`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`: cặp trùng `fedup2026` (`L229–L235`) vs. `qi2026fedup` (`L414–L425`); bản arXiv cũ `qi2023fedcspc` (`L119–L124`) & `agcl2023fedgcd` (`L215–L220`); 4 bài arXiv đang trích dẫn `guo2025fedorgp` (`L178–L183`), `lu2026gfpl` (`L407–L412`), `arivazhagan2019federated` (`L326–L331`), `liang2020think` (`L333–L338`); và 17 entry không trích dẫn.
- **Trích dẫn nguyên văn**:
  ```bibtex
  L229: @inproceedings{fedup2026, title={{FedUP}: Uncertainty-Aware Personalized Federated Learning via Probabilistic Prototypes}, ...}
  L414: @inproceedings{qi2026fedup, title = {FedUP: Uncertainty-Aware Personalized Federated Learning via Probabilistic Prototypes}, ...}
  L178: @article{guo2025fedorgp, ... journal = {arXiv preprint arXiv:2502.16119}, year = {2025} }
  ```
- **Phân tích chi tiết**:
  1. `fedup2026` (`L229–L235`, không được gọi nhưng có ngoặc `{FedUP}` đúng) và `qi2026fedup` (`L414–L425`, đang được gọi nhưng quên ngoặc `FedUP`) là hai bản ghi trùng lặp của cùng một bài báo IJCAI 2026.
  2. Trong số 17 entry chưa trích dẫn nằm trong `ieee2026.bib`, có 2 bản thảo `arXiv` thực chất đã được xuất bản tại hội nghị lớn (nếu tác giả định trích dẫn lại thì không được để bản arXiv cũ): `qi2023fedcspc` đã đăng tại *ACM MM 2023* (`pp. 3799–3807`) và `agcl2023fedgcd` đã đăng tại *IEEE/CVF CVPR 2024* (`pp. 28741–28750`, đồng thời dòng `L217` ghi thiếu 2 đồng tác giả và sai chính tả tên `Xingyuan Ji` thành `Xinyuan`).
  3. Bốn tài liệu arXiv đang được trích dẫn hợp lệ (`[12], [15], [18], [20]`) hiện khai báo dưới dạng `@article` với `journal = {arXiv preprint arXiv:...}`, làm cụm từ `arXiv preprint` bị in nghiêng như tên một tạp chí bình duyệt thay vì định dạng `@misc` / `note = {arXiv:XXXX.XXXXX}` theo hướng dẫn `IEEEtran_bst_HOWTO`.
- **Đề xuất cách sửa cụ thể**:
  Hợp nhất `fedup2026` vào `qi2026fedup`, dọn dẹp các entry không sử dụng (hoặc cập nhật `qi2023fedcspc`, `agcl2023fedgcd` lên bản chính thức nếu giữ lại), và chuyển 4 bài arXiv đang trích dẫn sang `@misc` với `howpublished = {arXiv:XXXX.XXXXX}` (hoặc `eprint`).

---

### `BIB-08` — Chuỗi Tên Hội nghị/Tạp chí Không Đồng nhất và Dư Trường `series`/`publisher` Giữa Các Mục ICML, ICLR và IEEE trong `Overleaf/ieee2026.bib`
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.bib`, `L99, L249–L250, L343, L364, L433–L439, L445–L450` (hiển thị tại `Overleaf/ieee2026.bbl`).
- **Trích dẫn nguyên văn**:
  ```latex
  % Overleaf/ieee2026.bbl:
  [17] ... in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306. PMLR, 2026, pp. 76 136–76 167.
  [19] ... in Proceedings of the International Conference on Machine Learning (ICML), 2021, pp. 2089–2099.
  [30] ... in Proceedings of the International Conference on Machine Learning (ICML), ser. PMLR, vol. 235, 2024, pp. 13 029–13 048.
  ```
- **Phân tích chi tiết**:
  Cùng hội nghị ICML nhưng xuất hiện dưới 3 định dạng hoàn toàn khác nhau trong danh mục tài liệu tham khảo; ICLR thiếu chữ `Proceedings of the`; và toàn bộ tên tạp chí/hội nghị được viết đầy đủ thay vì dùng tên viết tắt chuẩn IEEE (`IEEEabrv.bib`, ví dụ `IEEE Trans. Commun. Technol.`, `Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)`, `Proc. Int. Conf. Mach. Learn. (ICML)`, `Adv. Neural Inf. Process. Syst. (NeurIPS)`), kèm theo các trường thừa `publisher = {IEEE}` / `publisher = {PMLR}`.
- **Đề xuất cách sửa cụ thể**:
  Nạp thư viện viết tắt chuẩn `\bibliography{IEEEabrv,ieee2026}` hoặc chuẩn hóa đồng nhất chuỗi `booktitle` / `journal` theo chuẩn viết tắt IEEE và xóa các trường `publisher = {IEEE}` / `publisher = {PMLR}` không cần thiết.

---

### `VER-01` — Từ khóa `IEEEkeywords` Chưa Sắp xếp Theo Thứ tự Bảng Chữ cái (`L80–L82`) và Còn Sót Ghi chú Prompt Soạn thảo / Tên File Code Cũ / Comment Tiếng Việt trong Mã Nguồn `.tex`
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L80–L82` (`IEEEkeywords`), `L749–L750`, `L803–L804`, `L1561`, `L1592`.
- **Trích dẫn nguyên văn**:
  ```latex
  L80-L82:
  \begin{IEEEkeywords}
  Federated learning, prototype learning, class distributions, label skew, non-IID data.
  \end{IEEEkeywords}

  L749-L750: % Empirical paragraph: describe the curves in Figure 1 and state the
             % observation only if your experiment confirms it.
  L803-L804: % FedSCD Formulation with Literature & Codebase Mapping
             % Target Codebase: FLlib (clientnew_pcl.py & servernew_pcl.py)
  L1561:     % ==================== BẢNG 1: COMPONENT ABLATION ====================
  L1592:     % ==================== BẢNG 2: PARAMETERS & DESIGN CHOICES ====================
  ```
- **Phân tích chi tiết**:
  1. **Thứ tự `IEEEkeywords` trái quy tắc IEEE**: Theo *IEEE Editorial Style Manual* và `New_IEEEtran_how-to.tex` (`L139–L143`), các cụm từ khóa trong mục *Index Terms* (`IEEEkeywords`) bắt buộc phải được sắp xếp theo **thứ tự bảng chữ cái ABC (alphabetical order)**. Hiện tại danh sách xếp theo thứ tự ngẫu nhiên (`Federated...`, `prototype...`, `class...`, `label...`, `non-IID...`).
  2. **Comment nội bộ lộ diện khi nộp mã nguồn `.tex` cho IEEE**: Hệ thống xuất bản IEEE yêu cầu tác giả tải trực tiếp file nguồn `.tex` cho đội ngũ biên tập/chế bản. Hiện trong `Overleaf/ieee2026.tex` vẫn còn sót câu nhắc prompt soạn thảo (`L749–L750`), tên file mã nguồn cũ thời kỳ đầu (`clientnew_pcl.py & servernew_pcl.py` tại `L804`, trong khi tên thật hiện nay là `clientscd.py` và `serverscd.py`), và hai dòng comment tiếng Việt ghi lệch số thứ tự bảng (`BẢNG 1`, `BẢNG 2` tại `L1561, L1592` cho `Table II` và `Table III`).
- **Đề xuất cách sửa cụ thể**:
  - Sắp xếp lại `L80–L82` đúng thứ tự bảng chữ cái:
    ```latex
    \begin{IEEEkeywords}
    Class distributions, federated learning, label skew, non-IID data, prototype learning.
    \end{IEEEkeywords}
    ```
  - Xóa các dòng comment nháp `L749–L750`, `L802–L805` và đổi `L1561, L1592` sang tiếng Anh gọn chuẩn (`% Table II: Component Ablation`, `% Table III: Design Choices`).

---

## Phần D — Diễn đạt Tiếng Anh Học thuật, Ngữ pháp, Thuật ngữ/Từ viết tắt & Mạch Logic (17 Lỗi: `PROSE-01` – `LOGIC-06`)

> *Ghi chú khử trùng lặp mã định danh*: Các mã khuyết số thứ tự trong Phần D đã được hợp nhất vào đúng mục lỗi gốc để tránh trùng lặp: `PROSE-03` $\rightarrow$ **`IEEE-01`** (Phần C); `PROSE-05` $\rightarrow$ **`MATH-04`** (Phần A); `PROSE-09` & `STYLE-05` $\rightarrow$ **`LATEX-06`** (Phần C); `STYLE-01` $\rightarrow$ **`LATEX-03`** (Phần C); `STYLE-02` $\rightarrow$ **`LATEX-05`** (Phần C); `STYLE-03` & `STYLE-04` $\rightarrow$ **`FIG-05`** (Phần C) & **`PROSE-07`**; và `LOGIC-01` $\rightarrow$ **`EXP-01`** (Phần B).

### `PROSE-01` — Lỗi Ngữ pháp–Logic Câu So sánh Nhất trong Abstract (`"are highest among the six evaluated baselines"`), Thiếu Từ Nối Phản đề, Thuật ngữ `"diagonal second moments"` Chưa Chính xác và Giọng Văn Rào đón Phi Chuẩn IEEE
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L76–L78` (Abstract, dòng `L77`).
- **Trích dẫn nguyên văn**:
  ```latex
  Existing probabilistic and statistics-based methods already use distributional information for alignment, augmentation, and classifier estimation. We propose FedSCD (federated separable class distributions), which ... For fixed embeddings in common coordinates, raw pooling preserves their empirical first and diagonal second moments before numerical regularization and smoothing. ... Reported point estimates on four datasets under three Dirichlet label-skew settings are highest among the six evaluated baselines in all twelve settings. ... The evaluation concerns weighted local test accuracy with homogeneous backbones and full participation. Ablations support the tested combination of learning objectives and statistical references, with the empirical interpretation bounded by the reported evaluation protocol.
  ```
- **Phân tích chi tiết**:
  1. **Lỗi ngữ pháp-logic câu so sánh nhất + thiếu mạo từ `the`**: Cụm `"are highest among the six evaluated baselines"` vừa thiếu mạo từ xác định `"the"` trước tính từ so sánh nhất (`the highest`), vừa mắc lỗi logic tập hợp nghiêm trọng trong tiếng Anh: viết `X is the highest among Y` bắt buộc hàm ý $X \in Y$ (tức biến `FedSCD` thành 1 trong 6 baseline đang xét!). Để so sánh `FedSCD` với tập hợp 6 phương pháp đối chứng bên ngoài, phải dùng `"outperforms all six evaluated baselines"` hoặc `"achieves the highest accuracy compared with six evaluated baselines"`.
  2. **Đứt mạch chuyển ý từ công trình cũ sang đóng góp mới**: Câu thứ 2 nhượng bộ điều các phương pháp xác suất/thống kê hiện có đã làm được, rồi câu thứ 3 nhảy thẳng vào `"We propose FedSCD..."` mà thiếu hoàn toàn mệnh đề/liên từ phản đề (`However, ...`) chỉ ra khoảng trống kỹ thuật chưa được giải quyết.
  3. **Thuật ngữ xác suất chưa chuẩn xác**: Cụm `"diagonal second moments"` về mặt toán học chỉ moment bậc hai không quy tâm $\mathbb{E}[z^{\odot 2}]$, trong khi đại lượng được bảo toàn $(\sigma^c)^{\odot 2}$ là **moment trung tâm bậc hai** (*diagonal variances* / *second central moments*).
  4. **Văn phong biên bản tự kiểm điểm**: Các câu `"Reported point estimates..."`, `"The evaluation concerns..."`, và `"with the empirical interpretation bounded by the reported evaluation protocol"` nghe giống ghi chú miễn trừ trách nhiệm pháp lý hơn là văn phong súc tích, tự tin của một bài báo IEEE Transactions.
- **Đề xuất cách sửa cụ thể**:
  Viết lại đoạn Tóm tắt theo chuẩn IEEE Transactions (đồng thời xử lý luôn `TERM-01` và `TERM-03`):
  ```latex
  \begin{abstract}
  Prototype-based federated learning transfers class-level knowledge under heterogeneous client data, yet centroid separation alone fails to characterize intra-class feature dispersion. Although existing probabilistic and statistics-based methods leverage distributional cues for alignment, augmentation, or classifier estimation, jointly preserving cross-client dispersion during lightweight aggregation and regulating coordinate-wise local geometry without sharing model parameters remains challenging. We propose Federated Separable Class Distributions (FedSCD), which integrates shared class-conditional diagonal Gaussian statistics with local representation learning and classifier calibration under non-independent and identically distributed (non-IID) label skew. Client-adapted statistical anchors guide dispersion-aware angular objectives and signal-gated Mahalanobis compactness, while Gaussian virtual features calibrate separately optimized classifier heads. On the server, sample-weighted moment pooling accounts for both within-client dispersion and between-client mean disagreement, recovering exact empirical means and diagonal variances for fixed embeddings in common coordinates prior to numerical regularization and temporal smoothing. Each complete statistical message exchanges only $\mathcal{O}(Cd)$ scalars for $C$ classes and $d$ feature dimensions. Across four benchmarks and three Dirichlet label-skew settings under homogeneous backbones and full participation, FedSCD consistently outperforms all six evaluated baselines, reaching 76.66\% weighted local test accuracy on CIFAR-10 at $\alpha=0.5$ (+4.96 percentage points over FedTGP). Ablation studies confirm the complementary contributions of FedSCD's learning objectives and statistical references.
  \end{abstract}
  ```

---

### `PROSE-02` — Lỗi Đánh máy Thiếu Dấu Cách Sau Dấu Phẩy (`fluctuate,altering`) tại Dòng `L1190`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1190`, Section III-E (*Temporal smoothing of statistical references*).
- **Trích dẫn nguyên văn**:
  ```latex
  Consequently, single-round statistics may fluctuate,altering the references used for local learning.
  ```
- **Phân tích chi tiết**:
  Thiếu khoảng trắng sau dấu phẩy giữa hai từ `fluctuate` và `altering`, hiển thị trực tiếp thành chuỗi dính chữ `fluctuate,altering` trên bản PDF biên dịch (trang 8).
- **Đề xuất cách sửa cụ thể**:
  ```latex
  Consequently, single-round statistics may fluctuate across rounds, altering the references used for local learning.
  ```

---

### `PROSE-04` — Biệt ngữ Lập trình Phần mềm (`local statistics loader`, `statistics loader`, `The implementation`) và Mệnh đề Cụt (`; upload.`) trong Mục Phương pháp & `Algorithm 1`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L780–L782` (Section III-C), `L1207–L1208` (Section III-E), `L1316–L1317` & `L1347–L1348` (Section III-F, `Algorithm 1` & *Complexity*).
- **Trích dẫn nguyên văn**:
  ```latex
  L780-L782:   After local optimization, clients recompute $\{(n_k^c,\mu_k^c,\sigma_k^c)\}$ in a gradient-free pass through the local statistics loader and upload them to the server.
  L1207-L1208: The implementation smooths standard deviations directly, rather than variances.
  L1316-L1317: \STATE Recompute class-wise summaries from the statistics loader without gradients; upload.
  L1347-L1348: For a minibatch of size $B$, the current implementation constructs a full contrastive similarity matrix...
  ```
- **Phân tích chi tiết**:
  1. Cụm từ `"local statistics loader"` / `"statistics loader"` là tên biến đối tượng `DataLoader` trong PyTorch chứ không phải khái niệm toán học đã được định nghĩa ($\mathcal{D}_{k,t}^{\mathrm{stat}}$ ở `L592`).
  2. Cụm `"; upload."` ở cuối dòng 17 của `Algorithm 1` (`L1317`) là một động từ ngoại động bị cắt cụt thiếu cả tân ngữ lẫn đích nhận.
  3. Cách viết `"The implementation"` / `"the current implementation"` trong mục lý thuyết phương pháp đọc như tài liệu chú thích mã nguồn nội bộ thay vì trình bày thuật toán học thuật.
- **Đề xuất cách sửa cụ thể**:
  - `L780–L782`: `After local optimization, each participating client recomputes $\{(n_k^c,\mu_k^c,\sigma_k^c)\}$ in a gradient-free forward pass over $\mathcal{D}_{k,t}^{\mathrm{stat}}$ and uploads these tuples to the server.`
  - `L1207–L1208`: `FedSCD smooths coordinate-wise standard deviations directly rather than variances.`
  - `L1316–L1317`: `\STATE Recompute $\{(n_k^c,\mu_k^c,\sigma_k^c)\}_{c=1}^C$ over $\mathcal{D}_{k,t}^{\mathrm{stat}}$ without gradients, update local cache, and upload to the server.`
  - `L1347–L1348`: `For a minibatch of size $B$, evaluating the contrastive objectives via a full similarity matrix over valid samples and available class anchors requires at most $\mathcal{O}((B+C)^2d)$ operations.`

---

### `PROSE-06` — Các Cách Diễn đạt Thiếu Tự nhiên, Hoán dụ Nhân hóa, Khuyết Danh từ và Khuyết Mạo từ trong Phần Phân tích Thực nghiệm & Kết luận
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1499–L1500`, `L1514`, `L1519–L1520` (`Fig. 3` caption), `L1544`, và `L1657` (Sections V & VI).
- **Trích dẫn nguyên văn**:
  ```latex
  L1500:      Table~\ref{tab:com} summarizes the available accuracy point estimates for FedSCD and six baseline implementations...
  L1514:      However, Section IV evaluates each client on its own label distribution... FedGMKD has lower accuracy at the strongest than at the weakest skew on all four datasets.
  L1519-1520: Test-set-size-weighted local accuracy trajectories of FedSCD on four datasets at $\alpha=0.1$, over the displayed nominal round indices. ... The figure illustrates within-run learning behavior.
  L1544:      Toward the end of Fig.~\ref{fig:motivation}, FedCOF and FedSA exhibit larger margins than FedSCD...
  L1657:      Evaluation uses homogeneous backbones and full participation...
  ```
- **Phân tích chi tiết**:
  1. `"available accuracy point estimates"` (`L1500`) gây cảm giác nhóm tác giả chỉ thu thập lại được một vài file log rời rạc có sẵn thay vì tự thiết kế và thực hiện bộ thực nghiệm chuẩn.
  2. `"Section IV evaluates each client"` (`L1514`) dùng phép hoán dụ nhân hóa mục văn bản làm chủ thể hành động; còn `"at the strongest than at the weakest skew"` bị khuyết danh từ hoặc cấu trúc song song chuẩn sau tính từ `"strongest"`.
  3. `"over the displayed nominal round indices"` và câu `"The figure illustrates within-run learning behavior"` (`L1519–L1520`) là lối viết vòng vo, sáo rỗng không mang thông tin.
  4. `"Toward the end of Fig.~\ref{fig:motivation}"` (`L1544`) biến một hình vẽ tĩnh thành một quá trình thời gian (đúng ra là các vòng huấn luyện cuối trong Hình 1).
  5. `"Evaluation uses..."` (`L1657`) đứng đầu câu thiếu từ hạn định bắt buộc (`Our evaluation` hoặc `The present evaluation`).
- **Đề xuất cách sửa cụ thể**:
  - `L1500`: `Table~\ref{tab:com} reports the single-run weighted local test accuracy of FedSCD and six baseline methods across four benchmarks.`
  - `L1514`: `However, under the evaluation protocol in Section~\ref{sec:exp_setup}, each client is evaluated on its own local label distribution... Conversely, FedGMKD yields lower accuracy under the strongest skew ($\alpha=0.1$) than under the mildest skew ($\alpha=1.0$) across all four datasets.`
  - `L1519–L1520`: `Weighted local test accuracy trajectories of FedSCD across communication rounds on four datasets at $\alpha=0.1$.`
  - `L1544`: `In the late training rounds of Fig.~\ref{fig:motivation}, FedCOF and FedSA exhibit larger margins than FedSCD...`
  - `L1657`: `Our evaluation assumes homogeneous client backbones and full participation...`

---

### `PROSE-07` — Cụm Bổ ngữ Ghép Nhiều Gạch Nối Nặng nề, Trôi dạt Đơn vị (`percentage points` vs. `points`), Số Nhiều Chưa Chuẩn (`the numbers of ...`), Nối Lặp `and ... and` và Chuyển Đổi Chủ động–Bị động
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L207`, `L260–L261`, `L360–L361`, `L475`, `L1255–L1256`, `L1464`, `L1502`, `L1514`, `L1519`, `L1542`, `L1589`, `L1591`, `L1622`, `L1624`, `L1655`.
- **Trích dẫn nguyên văn**:
  ```latex
  L207, L1514, L1519, L1655: nearest-other-anchor Euclidean margins ... test-set-size-weighted local accuracy
  L260-L261, L475:           where $C$ and $d$ denote the numbers of classes and embedding dimensions, respectively.
  L360-L361:                 pFedFDA exchanges feature-extractor parameters and Gaussian distribution estimates and adapts generative classifiers...
  L1255-L1256:               Classes not yet observed use $\phi_t^c=\tau$, and the base temperature is used when no valid scores are available.
  L1502, L1589, L1591:       5.99 percentage points ... 1.64 and 0.37 points ... cross-entropy-plus-augmentation baseline
  ```
- **Phân tích chi tiết**:
  1. Các cụm tính từ ghép 3–4 gạch nối (`test-set-size-weighted`, `cross-entropy-plus-augmentation`, `nearest-other-anchor`) nặng nề không cần thiết trong khi chính bài báo đã có thuật ngữ chuẩn gọn hơn là `weighted local test accuracy` (`L77, L1458`) và `nearest-class Euclidean margin` (`L651`).
  2. Phần thực nghiệm (Section V, `L1502, L1589, L1591, L1622, L1624`) lúc viết đầy đủ `percentage points`, lúc rút thành `points` trần ngay trong cùng một đoạn văn.
  3. Cụm `the numbers of classes and embedding dimensions` (`L260–L261, L475`) dùng số nhiều `numbers` gượng ép; câu `L360–L361` nối liên tiếp hai từ `and`; và câu `L1255–L1256` chuyển đột ngột từ thể chủ động sang thể bị động giữa hai vế.
- **Đề xuất cách sửa cụ thể**:
  Dùng nhất quán `weighted local test accuracy` (định nghĩa một lần ở Section IV), `nearest-class Euclidean margin`, `the cross-entropy baseline with data augmentation`, và `percentage points` (hoặc định nghĩa viết tắt `pp` ở lần đầu trong Section V); sửa `L260–L261` thành `where $C$ denotes the number of classes and $d$ denotes the embedding dimension`; sửa `L360–L361` thành `Whereas pFedFDA exchanges both feature-extractor parameters and Gaussian estimates to adapt local generative classifiers, FedSCD...`; và sửa `L1255–L1256` thành `Unobserved classes ($c\notin\mathcal{C}_t$)—as well as all classes when $\mathcal{C}_t=\emptyset$—retain the base temperature $\phi_t^c=\tau$.`

---

### `PROSE-08` — Tiêu đề Mục Số Nhiều Phi Chuẩn (`\section{Experimental Evaluations}`, `L1492`)
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L1492` (Section V).
- **Trích dẫn nguyên văn**:
  ```latex
  \section{Experimental Evaluations}
  ```
- **Phân tích chi tiết**:
  Trong văn phong chuẩn của các tạp chí IEEE Transactions, danh từ trừu tượng `Evaluation` khi đứng làm tiêu đề mục luôn dùng ở dạng số ít (`Experimental Evaluation`), hoặc dùng danh từ đếm được `Experimental Results`.
- **Đề xuất cách sửa cụ thể**:
  Đổi `L1492` thành `\section{Experimental Results}` (hoặc `\section{Experimental Evaluation}`).

---

### `STYLE-06` — Cách Gọi Hình Con (`Panel (a)` Thay Vì `(a)`) ở Caption `Fig. 1` và Viết Hoa Không Đồng nhất `(vs. Shared)` ở `Table III`
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L207` (so với `L1524`) và `L1608` (`Table III`).
- **Trích dẫn nguyên văn**:
  ```latex
  L207:  Panel (a) reports nearest-other-anchor Euclidean margins, panel (b) reports ..., and panel (c) reports...
  L1608: \quad Class-specific projection (vs. Shared)            & 75.08 & -1.58 \\
  ```
- **Phân tích chi tiết**:
  Theo quy chuẩn chú thích hình của IEEE (`New_IEEEtran_how-to.tex`), các hình thành phần trong caption được ký hiệu trực tiếp là `(a)`, `(b)`, `(c)` thay vì diễn giải `Panel (a)` / `panel (b)`; đồng thời trong dòng bảng viết kiểu câu thường ở `L1608`, từ `Shared` trong `(vs. Shared)` bị viết hoa chữ cái đầu không cần thiết.
- **Đề xuất cách sửa cụ thể**:
  Tại `L207`, viết gọn: `(a) Nearest-class Euclidean margins, (b) dimension-rescaled Gaussian overlap score (Section~\ref{sec:exp_setup}), and (c) weighted local test accuracy.`; tại `L1608`, sửa thành `\quad Class-specific projection (vs. shared)`.

---

### `TERM-01` — Tên Phương pháp `FedSCD` Viết Ngược Thứ tự Giải nghĩa trong Abstract, Không Bao giờ Được Giải nghĩa trong Toàn bộ Thân Bài, Xuất hiện Trước Khi Giới thiệu ở `Fig. 1` (Cùng `FedCOF`/`FedSA` Chưa Trích dẫn) và Lẫn lộn 3 Kiểu Chữ
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L77` (Abstract), `L207` (`Fig. 1` caption), `L211–L214` (Section I), và `L797` (`Fig. 2` caption).
- **Trích dẫn nguyên văn**:
  ```latex
  L77:       We propose FedSCD (federated separable class distributions), which...
  L207:      ... as illustrated by FedCOF and FedSA relative to FedSCD toward the end of the displayed runs.
  L211-L214: Motivated by these observations, we propose \emph{FedSCD}, a communication-efficient...
  L797:      \caption{Overview of \textsc{FedSCD}. ...}
  ```
- **Phân tích chi tiết**:
  1. **Đảo ngược quy chuẩn viết tắt IEEE trong Abstract (`L77`)**: Chuẩn IEEE bắt buộc viết cụm từ đầy đủ trước rồi đặt từ viết tắt trong ngoặc đơn `Full Term (ACRONYM)`, không viết ngược `FedSCD (federated separable class distributions)`.
  2. **Quên giải nghĩa `FedSCD` trong toàn bộ phần Thân bài (`L84–L1658`)**: Abstract và Main Body là hai đơn vị độc lập về định nghĩa từ viết tắt theo quy định của IEEE, nhưng từ đầu Section I đến hết Section VI, tên đầy đủ *Federated Separable Class Distributions* không xuất hiện một lần nào!
  3. **Dùng trước khi giới thiệu ở `Fig. 1` (`L207`)**: Hình 1 và caption của nó nhắc tới `FedSCD` (cùng hai baseline `FedCOF` và `FedSA` chưa hề được giới thiệu hay gắn `\cite{...}`) trước đoạn văn giới thiệu `FedSCD` ở `L211–L214`.
  4. **Ba kiểu định dạng chữ khác nhau cho cùng một tên `FedSCD`**: In nghiêng `\emph{FedSCD}` tại `L212`, chữ hoa nhỏ `\textsc{FedSCD}` (`FEDSCD`) tại `L797`, và chữ đứng thường `FedSCD` ở các nơi còn lại.
- **Đề xuất cách sửa cụ thể**:
  - Tại Abstract (`L77`): `We propose Federated Separable Class Distributions (FedSCD), which...`
  - Tại Section I (`L211–L214`): `Motivated by these observations, we propose Federated Separable Class Distributions (FedSCD), a communication-efficient...` (bỏ `\emph`, và đổi `\textsc{FedSCD}` ở `L797` thành `FedSCD` đứng thường).
  - Tại caption `Fig. 1` (`L207`): bổ sung trích dẫn `FedCOF~\cite{goswami2025fedcof}` và `FedSA~\cite{zhou2025fedsa}` (và cân nhắc dời vị trí đặt khối `figure` xuống ngay sau đoạn `L211–L222`).

---

### `TERM-02` — Từ Viết tắt `EMA` Được Dùng 7 Lần Không Định nghĩa Xuyên suốt `Fig. 2` và Các Mục III–V Rồi Mới Được Giải nghĩa ở Tận Phần Kết luận (`L1653`)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L797` (`Fig. 2` caption), `L1189–L1212` (Section III-E), `L1437` (Section III-G), `L1453` & `L1464` (Section IV), `L1542`, `L1624` & `L1646` (Section V), và `L1653` (Section VI *Conclusion*).
- **Trích dẫn nguyên văn**:
  ```latex
  L797:  The server pools within-client variance and between-client scatter, applies EMA to means, standard deviations, and counts...
  ...
  L1653: ... before numerical regularization and exponential moving average (EMA) smoothing.
  ```
- **Phân tích chi tiết**:
  Từ viết tắt `EMA` xuất hiện liên tiếp 7 lần trong chú thích Hình 2 và các Mục III-G, IV, V mà chưa từng được giải nghĩa (ngay cả tiểu mục định nghĩa phương trình `(\ref{eq:ema})` tại Section III-E `L1189–L1212` cũng chỉ gọi chung là `"Temporal smoothing"` mà không hề nhắc tới từ `EMA`), rồi bất ngờ được mở rộng thành `exponential moving average (EMA)` ở tận đoạn đầu phần Kết luận (`L1653`)!
- **Đề xuất cách sửa cụ thể**:
  Định nghĩa chuẩn `exponential moving average (EMA)` ngay ở lần xuất hiện đầu tiên trong thân bài tại Section III-C (`L784`: `applies exponential moving average (EMA) temporal smoothing`), nhắc lại viết tắt `EMA` tại tiểu mục Section III-E (`L1191`), và ở Conclusion (`L1653`) chỉ dùng `EMA smoothing` không giải nghĩa lại.

---

### `TERM-03` — Các Từ Viết tắt Chưa Giải nghĩa Khi Dùng Lần Đầu (`non-IID` ở Abstract, `ProtoNCE`, `CNN`, `SGD`, `SupCon`, `CE`), Gán Nhầm Nhãn `SupCon` Cho $\mathcal{L}_{\mathrm{SC}}$ và Mở rộng Lại Thừa Từ `FL`
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L77` (Abstract), `L307` & `L574` (`federated learning`), `L508` (`ProtoNCE`), `L1447` (`CNN`), `L1452` (`SGD`), `L1577` (`SupCon`), `L1581` (`CE`).
- **Trích dẫn nguyên văn**:
  ```latex
  L77:   ... under non-IID label skew.
  L307:  ... prototype-based and statistical federated learning methods.
  L508:  FedSCD combines class-dependent ProtoNCE temperatures...
  L1447: All clients use the same CNN architecture...
  L1452: ... and SGD with a learning rate of 0.01...
  L1577: \quad w/o SupCon ($\lambda_{\text{ins}}=0$)
  L1581: \quad Baseline (CE + Augmentation only)
  ```
- **Phân tích chi tiết**:
  1. `SupCon` (`L1577`) và `CE` (`L1581`) xuất hiện đột ngột trong `Table II` mà chưa bao giờ được định nghĩa trong văn bản. Quan trọng hơn, việc gọi $\mathcal{L}_{\mathrm{SC}}$ là `SupCon` gây hiểu nhầm về mặt học thuật vì bài báo đã dày công phân biệt ở `L501–L516` và `L977–L978` rằng $\mathcal{L}_{\mathrm{SC}}$ là hàm mất mát tương phản lai mẫu–prototype đã loại bỏ các mẫu dương còn lại khỏi mẫu số (khác với SupCon kinh điển của Khosla et al.).
  2. `ProtoNCE` (`L508`) chưa từng được viết đầy đủ (`prototypical noise-contrastive estimation`) hoặc gắn trực tiếp với trích dẫn~\cite{li2021prototypical} ngay lần đầu xuất hiện; `CNN` (`L1447`), `SGD` (`L1452`) và `non-IID` trong Abstract (`L77`) cũng chưa được giải nghĩa lần đầu; trong khi `FL` đã được định nghĩa ở `L87` nhưng lại viết dài ra thành `federated learning` ở `L307` và `L574`.
- **Đề xuất cách sửa cụ thể**:
  Giải nghĩa `non-independent and identically distributed (non-IID)` trong Abstract (`L77`), `prototypical noise-contrastive estimation (ProtoNCE)~\cite{li2021prototypical}` tại `L508`, `cross-entropy (CE)` tại `L1029`, `convolutional neural network (CNN)` tại `L1447`, và `stochastic gradient descent (SGD)` tại `L1452`; dùng `FL` nhất quán tại `L307, L574`; và sửa nhãn dòng 2 của `Table II` (`L1577`) thành `\quad w/o Hybrid Contrastive ($\mathcal{L}_{\mathrm{SC}}, \lambda_{\mathrm{ins}}=0$)`.

---

### `TERM-04` — Trôi dạt Tên Gọi Các Thành phần Kỹ thuật Xuyên Suốt Bài báo và Nhãn Ngược Nghĩa `"No momentum (\beta=1.0)"` trong `Table III`
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L241` & `L291` vs. `L1119` & `L1134`; `L511` vs. `L965` vs. `L1453` vs. `L1577` vs. `L1589`; `L838` vs. `L1453` vs. `L1607`; `L1189` vs. `L1613–L1614` vs. `L1624`.
- **Trích dẫn nguyên văn**:
  ```latex
  % Cùng một cơ chế làm trơn nhưng đổi tên và ngược chiều tham số ở Table III (L1613-L1614):
  L1189: \paragraph{Temporal smoothing of statistical references.}
  L1195: \mu_t^c = (1-\beta)\mu_{t-1}^c + \beta\bar{\mu}_t^c
  L1613: \multicolumn{3}{l}{\textit{\textbf{Momentum Smoothing:}}} \\
  L1614: \quad No momentum ($\beta=1.0$)                         & 76.16 & -0.50 \\
  L1624: Disabling EMA gives 76.16\%, a reduction of 0.50 points.
  ```
- **Phân tích chi tiết**:
  1. **Mỗi mục gọi cùng một module bằng một tên khác nhau**:
     - Gom thống kê server: `drift-aware statistical aggregation` (`L241`) / `Drift-aware moment aggregation` (`L291`) $\leftrightarrow$ `Statistics-Consistent Aggregation` (`L1119`) / `Moment pooling with between-client scatter` (`L1134`, nơi từ `drift` biến mất hoàn toàn trong cả Section III-E!).
     - Hàm mất mát $\mathcal{L}_{\mathrm{SC}}$: `hybrid instance--prototype contrastive objective` (`L511`) $\leftrightarrow$ không đặt tên tại nơi định nghĩa (`L965`) $\leftrightarrow$ `instance contrastive weight` (`L1453`) $\leftrightarrow$ `SupCon` (`L1577`) $\leftrightarrow$ `hybrid supervised contrastive learning` (`L1589`).
     - Phép chiếu: `dispersion reweighting` (`L838`) $\leftrightarrow$ `dispersion gating` (`L1453`) $\leftrightarrow$ `Metric Projection` (`L1607`).
  2. **Nhãn `"No momentum (\beta=1.0)"` trong `Table III` (`L1613–L1614`) gây nhầm lẫn nghiêm trọng về chiều của $\beta$**: Trong phương trình `(20)` ($\mu_t^c = (1-\beta)\mu_{t-1}^c + \beta\bar{\mu}_t^c$), $\beta$ được định nghĩa là **trọng số của thống kê vòng hiện tại** (*EMA update rate*), ngược với hệ số quán tính `momentum` trong tối ưu hóa (nơi "no momentum" thường ứng với hệ số $= 0$, còn giữ lại lịch sử là $>0$). Gọi $\beta = 1.0$ là `"No momentum (\beta=1.0)"` khiến người đọc lướt bảng rất dễ hiểu ngược.
- **Đề xuất cách sửa cụ thể**:
  Chuẩn hóa một bộ tên gọi duy nhất xuyên suốt Section I, Section III, Hình 2 và `Tables II–III` (ví dụ: *Client-Adapted Anchor & Shared Dispersion-Signal Projection*, *Class-Temperature ProtoNCE* ($\mathcal{L}_{\mathrm{NCE}}$), *Hybrid Instance–Prototype Contrastive Loss* ($\mathcal{L}_{\mathrm{SC}}$), *Signal-Gated Mahalanobis Compactness* ($\mathcal{L}_{\mathrm{PL}}$), *Virtual-Feature Head Calibration* ($\mathcal{L}_{\mathrm{head}}$), và *Total-Variance Moment Pooling & EMA Smoothing*); đồng thời sửa `L1613–L1614` trong `Table III` thành:
  ```latex
  \multicolumn{3}{l}{\textit{\textbf{Temporal EMA Smoothing:}}} \\
  \quad Without EMA smoothing ($\beta=1.0$)               & 76.16 & $-0.50$ \\
  ```

---

### `TERM-05` — Dao động Từ vựng Phụ Cho Trọng số Số lượng Mẫu (`sample-weighted` vs. `sample-count-weighted` vs. `Count-weighted` vs. `count-pooled`) và Chưa Giải thích Chỉ số Dưới `PL` trong $\mathcal{L}_{\mathrm{PL}}$
- **Mức độ nghiêm trọng**: `Nitpick`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L77, L245, L442, L456, L991, L1142, L1379, L1464`.
- **Trích dẫn nguyên văn**:
  ```latex
  sample-weighted (L77, L245, L1142) | sample-count-weighted (L442) | Count-weighted (L1379) | count-pooled (L1464)
  \mathcal{L}_{\mathrm{PL}} = \frac{1}{B}\sum_{i\in\mathcal{I}} d^{y_i}(z_i).   % L991
  ```
- **Phân tích chi tiết**:
  Bốn biến thể từ vựng khác nhau cùng chỉ phép trung bình có trọng số theo $n_k^c$; đồng thời ký hiệu chỉ số dưới `PL` trong $\mathcal{L}_{\mathrm{PL}}$ (`L991`, viết tắt từ *Prototype Localization / Compactness* trong code cũ) không được giải thích nghĩa chữ cái khi giới thiệu hàm độ nén Mahalanobis.
- **Đề xuất cách sửa cụ thể**:
  Thống nhất dùng `sample-weighted` trên toàn bài, và chú thích ngắn gọn khi giới thiệu $\mathcal{L}_{\mathrm{PL}}$ ở `L991` (hoặc đổi ký hiệu thành $\mathcal{L}_{\mathrm{comp}}$).

---

### `LOGIC-02` — Giọng Điệu "Tự Phủ định / Miễn trừ Kiểm toán" (Defensive Audit-Disclaimer Tone) Dày đặc Làm Vụn Vỡ Mạch Thuyết phục Khoa học Xuyên suốt Bản thảo
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L77`, `L452–L453`, `L597–L601`, `L745–L748`, `L1013–L1014`, `L1209–L1211`, `L1237–L1239`, `L1436–L1437`, `L1504–L1511`, `L1529–L1532`, `L1546–L1555`, `L1626–L1646`.
- **Trích dẫn nguyên văn**:
  ```latex
  L452-L453:  This distinction concerns the combination and role of the mechanisms. Gaussian modeling and the law of total variance are established tools.
  L1509:      The comparison therefore supports examining the specific combination.
  L1529-1530: These trajectories describe the displayed runs. They do not establish monotonic optimization or a convergence rate...
  L1628-1630: Each removal changes a component within the full configuration, so the differences are conditional and cannot be added to attribute the overall improvement.
  ```
- **Phân tích chi tiết**:
  Để phản ứng lại các phê bình về tuyên bố quá mức (overclaiming) ở bản nháp trước, bản thảo hiện tại rơi sang thái cực ngược lại: gần như **cuối mỗi tiểu mục, mỗi hình vẽ và mỗi bảng kết quả đều chèn thêm 3–6 câu tự giới hạn mang tính phòng thủ cực đoan** (ví dụ: nhắc người đọc rằng định lý phương sai toàn phần là công cụ cổ điển đã biết, rằng đường cong thực nghiệm chỉ mô tả các lượt chạy đang hiển thị chứ không chứng minh định lý hội tụ đơn điệu, hay rằng các con số ablation loại trừ từng phần không được cộng tuyến tính với nhau). Việc rải dày đặc các câu rào đón hiển nhiên hoặc tự làm yếu luận điểm ngay tại Abstract, Related Work, Problem Setup và caption hình làm đứt gãy hoàn toàn nhịp đọc học thuật, tạo ấn tượng như văn bản được chắp vá tự động từ danh sách kiểm tra lỗi hơn là một công trình IEEE Transactions mạch lạc.
- **Đề xuất cách sửa cụ thể**:
  Giữ lại các phát biểu điều kiện toán học chính xác đúng **một lần** tại đoạn *Implications* cuối Section III-G (`L1436–L1437`), cắt bỏ toàn bộ các câu miễn trừ lặp vặt ở Abstract, Related Work, Problem Setup và các chú thích hình, sau đó gom tất cả các giới hạn thực nghiệm thực chất (xấp xỉ Gauss đường chéo, kiến trúc đồng nhất, tham gia đầy đủ, ước lượng đơn seed) vào **một tiểu mục ngắn gọn, chuẩn mực duy nhất `Discussion and Limitations`** ở cuối Section V (hoặc đoạn cuối Section VI).

---

### `LOGIC-03` — Lặp Nội dung Nặng nề ở Section II (*Related Work*) Khi Thuyết minh Lại Toàn bộ Kiến trúc FedSCD Trước Section III, Cùng Việc Lặp Lại Một Lưu ý 5–7 Lần
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L387–L397` (trùng `L136–L149`); `L442–L478` & `L497–L525` (trùng `L211–L261`, `L530–L558`, `L757–L787`); lưu ý về chỉ số chồng lấn co giãn lặp 5 lần (`L207, L727–L728, L1437, L1464, L1546`); lưu ý về bảo toàn moment trên tọa độ cố định lặp 7 lần (`L77, L249–L253, L460–L465, L1168–L1170, L1437, L1644–L1646, L1653–L1654`).
- **Trích dẫn nguyên văn**:
  ```latex
  % L474-L475 trong Related Work lặp nguyên văn từng chữ L258-L261 trong Introduction:
  Each complete class-wise message contains $\mathcal{O}(Cd)$ values, where $C$ and $d$ denote the numbers of classes and embedding dimensions, respectively.
  ```
- **Phân tích chi tiết**:
  Đúng như đồng tác giả `\NMD` đã chỉ ra ở `L398, L441, L496`: Tiểu mục II-B lặp lại nguyên đoạn lập luận động lực của phần Mở đầu; trong khi Tiểu mục II-C (dài 37 dòng cho riêng phần nói về FedSCD) và II-D (29 dòng) đi sâu mô tả lại toàn bộ quy trình gom phương sai server, làm trơn EMA, gói tin tải lên/phát xuống, độ phức tạp $\mathcal{O}(Cd)$, cách tính nhiệt độ theo lớp và cấu trúc mẫu số của hàm tương phản trước cả khi người đọc bước vào Section III (*Proposed Method*)! Bên cạnh đó, câu giải thích sự khác nhau giữa điểm chồng lấn co giãn theo $\sqrt{d}$ và `Eq. (4)` bị nhắc đi nhắc lại **5 lần**, còn điều kiện bảo toàn moment thực nghiệm trước khi làm trơn bị nhắc lại tới **7 lần** từ Abstract tới Conclusion.
- **Đề xuất cách sửa cụ thể**:
  Thu gọn phần định vị cuối mỗi tiểu mục của Section II (`L387–L408`, `L442–L478`, `L497–L525`) xuống còn **1 đoạn ngắn 4–5 câu**, chỉ tập trung đối sánh điểm khác biệt cốt lõi giữa FedSCD với các nhóm công trình vừa nêu (`FedTGP/FedSA`, `FedUP/FedCDWA/GFPL/FedCOF`, `SupCon/PCL`) và nhường toàn bộ chi tiết cơ chế cho Section III. Chỉ định nghĩa lưu ý chỉ số chồng lấn $\exp(-D_B^{\mathrm{eval}}/\sqrt{d})$ một lần tại Section IV (`L1464`) và điều kiện bảo toàn moment một lần tại Proposition~\ref{prop:exact} / Section III-G.

---

### `LOGIC-04` — Chưa Giải thích Cơ chế Căn chỉnh Tọa độ Đặc trưng Liên Client Khi Không Chia sẻ Trọng số Backbone (`L599–L601` vs. Section IV) và Khoảng trống Giữa Tuyên bố Đóng góp 3 với Thực nghiệm Ablation ($B_t^c = 0$)
- **Mức độ nghiêm trọng**: `Major`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L241–L253` & `L291–L300` (Section I), `L582–L601` (Section III-A), `L1128–L1171` (Section III-E), `L1446–L1455` (Section IV), và `L1634–L1638` (Section V-C).
- **Trích dẫn nguyên văn**:
  ```latex
  L599-L601:  The statistical interpretation of cross-client pooling assumes comparable feature coordinates. A shared dimension alone does not establish semantic coordinate alignment.
  L1634-1635: The reported ablations do not separately remove the between-client scatter term while retaining the remaining aggregation and learning mechanisms.
  ```
- **Phân tích chi tiết**:
  1. **Câu hỏi cốt tử của phản biện về tính hoán vị/căn chỉnh tọa độ (Coordinate Alignment)**: Vì một mạng nơ-ron có tính bất biến hoán vị nút ẩn và FedSCD **không bao giờ đồng bộ tham số bộ trích xuất $\theta_{f_k}$** (`L588–L589`), phản biện sẽ hỏi ngay: *tại sao tọa độ thứ $j \in \{1,\dots,d\}$ của client 1 lại mang cùng ý nghĩa với tọa độ thứ $j$ của client 2 để có thể cộng/gom trung bình $\mu_k^c$ và phương sai $(\sigma_k^c)^{\odot 2}$ theo từng chiều?* Dòng `L599–L601` nêu thẳng cảnh báo rằng cùng số chiều $d$ không tự tạo ra căn chỉnh tọa độ, rồi bỏ lửng hoàn toàn mà không hề giải thích rằng trong thực tế: (i) tất cả client xuất phát từ **cùng một điểm khởi tạo trọng số chung $(\theta_f^{(0)}, \theta_h^{(0)})$** (`ALGO-01`), và (ii) ngay sau vòng đầu tiên, các mỏ neo toàn cục $\mu^c$ cùng hàm phạt độ nén tọa độ $d^c(z)$ đóng vai trò hệ trục neo chung giữ cho các tọa độ không trôi lệch hoán vị.
  2. **Khoảng trống bằng chứng cho Trụ cột 2 / Đóng góp 3 ($B_t^c$)**: Phần Mở đầu (`L241, L291`), Section III-E (`L1128–L1129`) và Proposition~\ref{prop:drift} nhấn mạnh vai trò của số hạng độ lệch trung bình liên client $B_t^c$ trong việc ngăn chặn trọng số nghịch đảo phương sai quá lớn và phân phối đặc trưng ảo quá hẹp. Tuy nhiên phần thực nghiệm lại không có phép thử tắt riêng $B_t^c = 0$ (chỉ trung bình hóa $(\sigma_k^c)^{\odot 2}$), buộc cuối Section V-C (`L1634–L1638`) phải thừa nhận tác động độc lập của $B_t^c$ lên độ chính xác vẫn là câu hỏi mở.
- **Đề xuất cách sửa cụ thể**:
  Bổ sung lời giải thích tường minh về khởi tạo trọng số đồng nhất $(\theta_f^{(0)}, \theta_h^{(0)})$ kết hợp lực neo từng tọa độ của $(\mu^c, \sigma^c)$ tại Section III-A (`L599–L601`) và Section IV (`L1447`); đồng thời chạy bổ sung 1 lượt ablation tắt $B_t^c = 0$ trên CIFAR-10 ($\alpha=0.5$) đưa vào `Table III` (chỉ cần sửa 1 dòng trong `serverscd.py:198` bỏ `diff_sq`) để hoàn thiện trọn vẹn chuỗi lập luận từ Lý thuyết tới Thực nghiệm.

---

### `LOGIC-05` — Xung đột Mạch Dẫn chuyện Khi Dùng `FedTGP` Làm Ví dụ Điển hình Cho Chồng lấn Phân phối Cao ở Văn bản Nhưng `Fig. 1` Lại Cho Thấy `FedTGP` Có Độ Chồng lấn Thấp Hơn Cả `FedSCD`, Cùng `Fig. 3` Đơn Điệu Không Có Đường Cong Baseline
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L132–L149` & `L207` (Section I), `L647–L659` & `L718–L726` (Section III-B), và `L1516–L1532` (`Fig. 3`, Section V-A).
- **Trích dẫn nguyên văn**:
  ```latex
  L647-L659: Prototype-based methods such as FedTGP~\citep{zhang2024fedtgp} promote discrimination by separating class prototypes. ... However, this point-based criterion ignores within-class feature dispersion ... so well-separated prototypes need not imply limited inter-class overlap.
  L207:      FedTGP has lower overlap than FedSCD while also having lower accuracy.
  ```
- **Phân tích chi tiết**:
  1. Cả Section I (`L132–L149`) lẫn mở đầu Section III-B (`L647–L659`) đều lấy đích danh `FedTGP` làm đối tượng dẫn dắt cho luận điểm *"tách rời tâm cụm không ngăn được sự chồng lấn phân phối lớn"*. Thế nhưng ngay khi nhìn vào `Fig. 1(b)` (`L207, L721`), độc giả lại thấy **`FedTGP` có điểm chồng lấn thấp nhất — thấp hơn cả `FedSCD`** (trong khi `FedSA` và `FedCOF` mới là hai phương pháp có biên lớn nhưng chồng lấn cao)! Dù tác giả đã viết câu chữa cháy ở `L207, L721–L726`, cách dẫn dắt ban đầu vẫn tạo cảm giác mẫu thuẫn giữa giả thuyết mở đầu và hình minh họa.
  2. `Figure 3` (`L1516–L1521`) chỉ vẽ 4 đường độ chính xác của riêng mình `FedSCD` trên 4 tập dữ liệu ở $\alpha=0.1$ mà không vẽ kèm bất kỳ baseline đối chứng nào (như `FedCOF` hay `FedTGP`), mang lại rất ít thông tin so sánh khoa học (đúng như phê bình của đồng tác giả `\NMD` tại `L1532`).
- **Đề xuất cách sửa cụ thể**:
  - Cấu trúc lại luận điểm dẫn dắt ở Section I và III-B thành **hai hạn chế bổ sung nhau** được minh chứng trực tiếp bởi `Fig. 1`: (i) tối đa hóa khoảng cách mỏ neo mà bỏ qua kiểm soát phương sai dẫn tới chồng lấn phân phối cao (như `FedSA`, `FedCOF`), và (ii) chỉ ép tách xa các điểm prototype toàn cục mà thiếu hiệu chuẩn phân phối cho đầu phân loại cục bộ dưới lệch nhãn (như `FedTGP`, `FedORGP`) thì dù giảm chồng lấn gộp vẫn cho độ chính xác phân loại cục bộ thấp.
  - Bổ sung đường cong của 2–3 baseline cạnh tranh mạnh nhất (`FedCOF`, `FedTGP`, `FedORGP`) vào `Fig. 3` (chia thành lưới $2\times 2$ cho 4 tập dữ liệu, nơi sự sụp đổ của `FedTGP` trên Flowers102 cũng sẽ hiện lên cực kỳ thuyết phục!), hoặc thay `Fig. 3` bằng đồ thị độ nhạy siêu tham số.

---

### `LOGIC-06` — Đảo Ngược Trình tự Diễn giải trong Section III-A (Nhắc Tới `"the gradient-free statistics pass"` và Quy tắc Loại bỏ Minibatch Cuối ở `L591–L598` Trước Khi Giới thiệu Quá trình Huấn luyện & Trích xuất ở `L605–L607`)
- **Mức độ nghiêm trọng**: `Minor`
- **Vị trí chính xác**: `Overleaf/ieee2026.tex`, `L591–L610`, Section III-A (*Problem Setup and Notation*).
- **Trích dẫn nguyên văn**:
  ```latex
  For statistical communication in round $t$, let $\mathcal{D}_{k,t}^{\mathrm{stat}}\subseteq\mathcal{D}_k$ contain the samples processed in the gradient-free statistics pass. ... These counts refer to processed samples, including the incomplete-minibatch exclusion specified in Section IV. ... After local optimization, each participating client extracts unnormalized embeddings in evaluation mode without gradient computation.
  ```
- **Phân tích chi tiết**:
  Ngay tại `L593`, bài báo dùng mạo từ xác định `"the gradient-free statistics pass"` như thể người đọc đã biết bước này là gì, đồng thời đưa chi tiết cài đặt `drop_last` ở `L597–L598` vào trước cả khi giới thiệu rằng sau bước tối ưu cục bộ, client thực hiện một lượt trích xuất đặc trưng không tính gradient ở chế độ evaluation (`L605–L607`).
- **Đề xuất cách sửa cụ thể**:
  Đảo lại trật tự tự nhiên tại `L591–L610`: giới thiệu trước rằng sau khi hoàn tất cập nhật cục bộ ở vòng $t$, mỗi client tham gia thực hiện một lượt lan truyền thuận không tính gradient ở chế độ đánh giá trên tập con mẫu hợp lệ $\mathcal{D}_{k,t}^{\mathrm{stat}} \subseteq \mathcal{D}_k$ (chuyển chi tiết loại bỏ minibatch lẻ sang `Section~\ref{sec:exp_setup}`), rồi mới định nghĩa $\mathcal{D}_k^c$, $n_k^c$, $\mu_k^c$ và $(\sigma_k^c)^{\odot 2}$ ở `Eqs. (1)–(2)`.

---

## 4. Checklist Thứ tự Ưu tiên Chỉnh sửa (Giai đoạn 1 → Giai đoạn 2 → Giai đoạn 3)

Để giúp nhóm tác giả triển khai chỉnh sửa an toàn, có hệ thống và không phát sinh xung đột chéo sau khi duyệt báo cáo kiểm toán này, toàn bộ **70 lỗi** được phân bổ vào 3 giai đoạn thực thi theo thứ tự ưu tiên giảm dần:

### Giai đoạn 1 — Khẩn cấp: Chặn Nguy cơ Desk-Reject & Khôi phục Tính Toàn vẹn Học thuật / Thực nghiệm (9 Lỗi `Critical`)
*Mục tiêu: Giải quyết triệt để mọi lỗi gây loại bài tự động về mặt thủ tục/học thuật và thống nhất nguồn sự thật (single source of truth) giữa số liệu bài báo, hình vẽ và mã nguồn.*

- [ ] **1.1. Điền thông tin thật của tác giả & tạp chí mục tiêu (`IEEE-02`)**: Thay thế toàn bộ placeholder `Firstname1 Lastname`, `Department of XXX, University of YYY` và `\markboth{IEEE Transactions on ...}` tại `Overleaf/ieee2026.tex` (`L67–L73`).
- [ ] **1.2. Dọn sạch 100% cờ nháp, macro bình luận và chữ tô đỏ (`IEEE-01`)**: Xóa `-New-12/11` và `\textcolor{red}{New 13/11}` khỏi 7 tiêu đề mục; xử lý và xóa cả 5 khối `\NMD{...}`; gỡ toàn bộ ~35 lệnh `\textcolor{red}{...}` về chữ đen chuẩn; xóa `L54–L60`.
- [ ] **1.3. Loại bỏ và sửa chữa 2 tài liệu tham khảo bịa đặt / sai tác giả (`BIB-01`, `BIB-02`)**: Xóa ngay `chen2025hproto` (`[10]`, thay thế trích dẫn tại `L115` bằng bài báo có thật) và `liu2025fedpeft` khỏi `Overleaf/ieee2026.bib`; sửa lại đúng tiêu đề và 3 tác giả thật (`Oh, Kim, Yun`) cho `oh2022fedbabu` (`[21]`).
- [ ] **1.4. Khắc phục công bằng augmentation & đối xứng vòng lặp đánh giá (`CODE-01`, `EXP-02`)**:
  - Chuẩn hóa vòng lặp `serverscd.py` về đúng 100 vòng huấn luyện và đánh giá đồng pha với 6 baseline (hoặc công bố rõ độ lệch chỉ số vòng `1..101` vs. `0..100` và hiệu chỉnh trục hoành `Fig. 1`, `Fig. 3`).
  - Bật cùng phép augmentation ảnh (flip + reflection crop) cho cả 6 baseline và chạy lại `Table I` để giải quyết triệt để nghịch lý `Table II` (`72.79%`) > `Table I` (`71.70%`) (hoặc công bố minh bạch sự khác biệt augmentation + bổ sung dòng `w/o Input Augmentation` vào `Table II`).
- [ ] **1.5. Đồng bộ 100% siêu tham số giữa bài báo và mã nguồn (`CODE-02`)**: Xác nhận bộ siêu tham số tạo ra kết quả `76.66%` ($\beta, \lambda, \lambda_{\mathrm{ins}}, \lambda_{\mathrm{vl}}, \lambda_{\mathrm{vg}}$) và cập nhật đồng bộ giữa `L1453`, `main.py`, `clientscd.py`, `serverscd.py`, các script chạy và `ABLATION_RUN_GUIDE.md`.
- [ ] **1.6. Minh bạch hóa chỉ số `Table I` & xóa tham chiếu treo (`EXP-01`)**: Công bố rõ quy tắc chọn điểm đánh giá (`max_acc` hay vòng cuối, khuyến nghị báo cáo cả hai kèm chú thích các run `FedTGP`/`FedPLVM` bị phân kỳ/crash) tại Section IV, và xóa các tham chiếu đứt gãy tới *"limitations in Section IV"* ở `L1500, L1531–L1532`.
- [ ] **1.7. Vẽ và xuất lại sơ đồ tổng quan `Overleaf/figures/framework.pdf` (`FIG-01`)**: Sửa $\mathcal{N}(p^c, \mathrm{diag}((\eta_b\widetilde{\sigma}^c)^{\odot 2}))$ + $\Pi(\cdot)$, sửa công thức EMA, đổi $\phi_t^c \to \phi^c$ (hoặc $\phi_{t-1}^c$) ở phía client, thêm nhãn $\mu^c$, thêm mũi tên $z \to h_k$ không chặn gradient cho $\ell_{\mathrm{CE}}(h_k(z),y)$, và đổi `\textsc{FedSCD}` thành `FedSCD` ở `L797`.

---

### Giai đoạn 2 — Cốt lõi: Chuẩn hóa Toán học, Pseudocode, Bổ sung Thực nghiệm, Cấu trúc `IEEEtran` & Mạch Lập luận (33 Lỗi `Major`)
*Mục tiêu: Đảm bảo mọi phương trình/thuật toán tự đóng kín, tái lập được 100%, chuẩn hóa bố cục `IEEEtran` và nâng tầm sức thuyết phục trước phản biện chuyên sâu.*

- [ ] **2.1. Hoàn thiện chặt chẽ hệ thống phương trình & hằng số (`MATH-01`..`MATH-05`, `CODE-03`, `CODE-05`)**:
  - Điền đủ 7 hằng số vào Section IV ($\kappa=10, \epsilon_\mu=10^{-8}, \epsilon_\sigma=10^{-4}, \epsilon_v=10^{-8}, \epsilon_s=\epsilon_r=10^{-6}, \epsilon_g=10^{-8}, \epsilon_{\mathrm{norm}}=10^{-12}$) kèm ràng buộc miền dương (`MATH-01`).
  - Đối chiếu và sửa hằng số co cụm $n_0 = 50$ khớp `clientscd.py:185` (`CODE-03`), bổ sung chi tiết `ReLU` cuối encoder ($z \in \mathbb{R}_{\ge 0}^d$), tiền xử lý ảnh $32\times 32$ / $64\times 64$ và cấu hình `DataLoader` (`CODE-05`).
  - Tách phương sai thực nghiệm thô $(s_k^c)^{\odot 2}$ với $\sigma_k^c$ tải lên có fallback $10^{-6}\mathbf{1}$ khi $n_k^c=1$ và đệm không khi $n_k^c=0$ (`MATH-02`).
  - Viết lại $\mathcal{L}_{\mathrm{SC}}$ bằng họ chỉ số hợp rời $\mathcal{R}_i = (\mathcal{I}\setminus\{i\})\sqcup\mathcal{A}$, tránh trùng ký hiệu $p, n, j$ (`MATH-03`).
  - Đánh chỉ số nhiễu $\zeta_i^{\mathrm{loc}}, \zeta^{c,\mathrm{glob}}$, đưa hàm chỉ báo $\mathbb{I}_{k,t}^{\mathrm{stat}}$ bảo vệ mẫu số vào `Eq. (17)` và đóng kín biểu thức $\lambda_{k,t}$ (`MATH-04`).
  - Đưa công thức chính xác của $D_B^{\mathrm{eval}}(c,c')$ (khớp `motivation_metrics.py`) và định nghĩa toán học của biến thể `"Class-specific projection"` vào Section IV & V-C (`MATH-05`).
- [ ] **2.2. Chuẩn hóa `Algorithm 1`, độ phức tạp và bảng Ablation/Sensitivity (`ALGO-01`..`04`, `EXP-03`, `EXP-04`, `EXP-06`)**:
  - Bổ sung `\ENSURE`, dữ liệu $\{\mathcal{D}_k\}$, khởi tạo chung $(\theta_f^{(0)}, \theta_h^{(0)})$ và lưu cache $(\breve{n}_k^c, \breve{\mu}_k^c)$ (`ALGO-01`); đưa phép chiếu mẫu $\hat{z}_i$ vào trong vòng lặp minibatch, ghi rõ $\nabla_{\theta_{h_k}}\mathcal{L}_{\mathrm{head}}$ vs. $\nabla_{\theta_{f_k}}\mathcal{L}_{\mathrm{enc}}$ và tính đủ 2 lượt truyền thuận qua $h_k$ trong mục Complexity (`ALGO-02`).
  - Tách điều kiện bật đặc trưng ảo khỏi $\lambda > 0$ trong `Algorithm 1` / `clientscd.py` (`ALGO-03`) và thêm điều kiện `for each class c with \bar{n}_t^c > 0` ở bước gom moment server (`ALGO-04`).
  - Đặt lại tên chính xác cho dòng cơ sở cục bộ trong `Table II`, bổ sung các dòng ablation cốt lõi đặc biệt là `w/o Between-Client Scatter ($B_t^c = 0$)` (`EXP-03`, `LOGIC-04`); bổ sung chạy đa seed (`mean ± std`) và quét siêu tham số hoặc đổi tên `Table III` (`EXP-04`); công bố việc `FedPLVM` và `FedGMKD` gom toàn bộ trọng số mạng $\mathcal{O}(|\theta|)$, bổ sung siêu tham số của 6 baseline và nhận xét thêm về `FedORGP` (`EXP-06`).
- [ ] **2.3. Chuẩn hóa định dạng `IEEEtran`, LaTeX, `Fig. 1` và BibTeX (`IEEE-03`..`05`, `LATEX-01`..`05`, `FIG-03`, `BIB-03`..`05`)**:
  - Đổi `\paragraph{Heading.}` thành `\subsubsection{Heading}` không dấu chấm cuối (`IEEE-03`); bỏ `[section]` để đánh số `Proposition 1–3` liên tục (`IEEE-04`); gỡ các gói xung đột (`enumitem`, `todonotes`, `float`, `\marginparwidth`), bật `balance`, và xóa 5 dòng trống bên trong `itemize` ở `L265–L314` (`IEEE-05`).
  - Bỏ `\input{math_commands.tex}` (`LATEX-01`), thay `\citep`/`\citet` bằng `~\cite` chuẩn (`LATEX-02`), thêm `\label{sec:exp_setup}` cho Section IV và cập nhật 10 tham chiếu (`LATEX-03`), xóa `\vspace{-10pt}` ở `L1491` và `\vspace{1.5mm}` ở `Tables II–III` (`LATEX-04`), chuyển 5 dòng phụ `(6)–(10)` sang `align*`, đánh số cho $\widetilde{z}^{c,b}$ và $\bar{\sigma}_t^c$, bỏ dấu hai chấm trước phương trình (`LATEX-05`).
  - Vẽ lại `Fig. 1` ở kích thước chuẩn 1 cột với font `8–9pt`, thêm nhãn trục tung cả 3 panel và bổ sung/chú thích `FedPLVM` (`FIG-03`).
  - Xóa mảnh BibTeX cụt đầu `L380–L385` (`BIB-03`), bọc ngoặc `{GFPL}`, `{FedUP}`, `{FedCDWA}`, `{Wasserstein}` (`BIB-04`), điền đủ 5 tác giả cho `lu2026gfpl` và sửa `number = {1--2}` cùng danh sách tác giả cho `kairouz2021advances` (`BIB-05`).
- [ ] **2.4. Tái cấu trúc diễn đạt Abstract, hệ thống từ viết tắt/thuật ngữ và mạch văn (`PROSE-01`, `TERM-01`, `TERM-02`, `TERM-04`, `LOGIC-02`..`04`)**:
  - Viết lại Abstract chuẩn ngữ pháp và văn phong IEEE (`PROSE-01`); chuẩn hóa quy tắc giải nghĩa `FedSCD` và `EMA` ngay lần đầu xuất hiện ở Abstract & Main Body (`TERM-01`, `TERM-02`); đồng bộ tên các module kỹ thuật và sửa nhãn `"Without EMA smoothing (\beta=1.0)"` ở `Table III` (`TERM-04`).
  - Cắt gọn phần lặp kiến trúc FedSCD trong Section II (`LOGIC-03`), lược bỏ các câu tự rào đón vụn vặt để gom về một tiểu mục `Discussion and Limitations` duy nhất cuối Section V (`LOGIC-02`), và giải thích rõ cơ chế căn chỉnh tọa độ nhờ khởi tạo trọng số chung $(\theta_f^{(0)}, \theta_h^{(0)})$ (`LOGIC-04`).

---

### Giai đoạn 3 — Hoàn thiện Xuất bản: Tinh chỉnh Ký hiệu Phụ, Bảng biểu, BibTeX Chi tiết & Trau chuốt Câu chữ (21 Lỗi `Minor` + 7 Lỗi `Nitpick`)
*Mục tiêu: Đạt chất lượng chế bản hoàn hảo (camera-ready polish) với 0 cảnh báo biên dịch LaTeX/BibTeX.*

- [ ] **3.1. Tinh chỉnh ký hiệu toán học & chi tiết giao thức nhỏ (`MATH-06`..`09`, `ALGO-05`, `CODE-06`, `EXP-05`)**:
  - Khử trùng lặp các ký hiệu phụ ($\psi^c(z)$ thay cho $d^c(z)$, $[z]_j$ cho tọa độ, $\nu_{k,j}^c$ thay cho $m_j^c$, $\bar{\sigma}_{c,c',j}^2$) (`MATH-06`); phân biệt ký hiệu cache vòng trước $(\breve{\mu}_k^c, \breve{n}_k^c)$ và khai báo việc ẩn chỉ số $k$ (`MATH-07`); dùng $\mathbf{1}$, $\oslash$ và $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ nhất quán (`MATH-08`); đồng bộ `\ln`, viết tường minh $\mathcal{L}_{\mathrm{mix}}$ và định nghĩa ngắn gọn `\operatorname{sg}`, `\operatorname{clip}`, `\operatorname{round}` (`MATH-09`).
  - Thống nhất mô tả gói phát sóng downlink $C(2d+1)$ vs. $C(2d+2)$ (`ALGO-05`); chú thích chế độ trích xuất thống kê của `FedSA` và sửa mặc định `--lamda_proto` trong code (`CODE-06`); cập nhật đúng 5 con số làm tròn (`39.98, 56.38, 61.20, 40.86, 20.34`) trong `Table I` (`EXP-05`).
- [ ] **3.2. Dọn sạch cảnh báo biên dịch, bảng biểu, file thừa & metadata BibTeX (`IEEE-06`..`07`, `LATEX-06`..`07`, `FIG-02`, `FIG-04`..`05`, `BIB-06`..`08`, `VER-01`)**:
  - Đổi `lettersize` thành `letterpaper` ở `L1` (`IEEE-06`); đổi `[t]` thành `[!t]` cho cả 6 float (`IEEE-07`); xóa khoảng trắng xuống dòng trước `~\cite`, bỏ ngoặc `{=}` và chuẩn hóa `\mathrm{...}` (`LATEX-06`); dọn các file `.aux` xung đột Google Drive, `missfont.log`, file `.md` nội bộ và file hình mồ côi `ieee_tsne_cifar10_dir05.pdf`, rồi chạy đủ 4 lượt `latexmk -pdf` (`LATEX-07`, `FIG-02`, `FIG-04`).
  - Thay `\resizebox` ở `Table I` bằng `\footnotesize` + `tabular*`, sửa `21.8` $\rightarrow$ `21.80`, `-0.9` $\rightarrow$ `$-0.90$`, đổi dấu trừ sang chế độ toán `$...$`, và đồng bộ Title Case cho cả 3 caption bảng (`FIG-05`).
  - Bổ sung `volume`/`pages` còn thiếu cho 5 bài hội nghị (`BIB-06`), gộp entry trùng `fedup2026`, dọn 17 entry không trích dẫn và chuẩn hóa định dạng arXiv/tên viết tắt hội nghị (`BIB-07`, `BIB-08`); sắp xếp `IEEEkeywords` theo thứ tự ABC và xóa các comment nháp/tiếng Việt trong `.tex` (`VER-01`).
- [ ] **3.3. Trau chuốt văn phong tiếng Anh & trật tự trình bày (`PROSE-02`, `PROSE-04`, `PROSE-06`..`08`, `STYLE-06`, `TERM-03`, `TERM-05`, `LOGIC-05`..`06`)**:
  - Sửa lỗi dính chữ `fluctuate, altering` (`PROSE-02`), thay thế biệt ngữ `statistics loader` / `The implementation` (`PROSE-04`), sửa các câu hoán dụ/khuyết danh từ/mạo từ ở Section V–VI (`PROSE-06`), đơn giản hóa các cụm ghép nhiều gạch nối và thống nhất đơn vị `percentage points` (`PROSE-07`), đổi tiêu đề Section V thành `Experimental Results` (`PROSE-08`), dùng `(a), (b), (c)` và `(vs. shared)` (`STYLE-06`).
  - Giải nghĩa lần đầu `non-IID`, `ProtoNCE`, `CNN`, `SGD`, `CE`, sửa nhãn `w/o Hybrid Contrastive` ở `Table II` (`TERM-03`), thống nhất thuật ngữ `sample-weighted` (`TERM-05`), điều chỉnh mạch dẫn chuyện `FedTGP` vs. `FedSA`/`FedCOF` kèm nâng cấp `Fig. 3` (`LOGIC-05`), và sắp xếp lại trình tự giới thiệu lượt trích xuất thống kê ở đầu Section III-A (`LOGIC-06`).

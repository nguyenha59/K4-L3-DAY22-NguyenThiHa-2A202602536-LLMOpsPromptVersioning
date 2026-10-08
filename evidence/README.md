# Evidence — Day 22: LangSmith + Prompt Versioning

## Danh sách file

| File | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Lọc tag `rag`: 50 trace `rag-query` trên LangSmith (Bước 1) |
| `01_langsmith_traces_all.png` | Toàn bộ project `day22-lab`: 402 trace (≥ 100 — gồm `rag-query`, `ab-rag-query` và các run RAGAS) |
| `02_prompt_hub.png` | 2 prompt `nguyen-ha-rag-prompt-v1` / `nguyen-ha-rag-prompt-v2` trên Prompt Hub |
| `02_ab_routing_log.txt` | Log A/B routing: pull 2 prompt từ Hub, nhãn `[prompt-v1]` / `[prompt-v2]` cho 50 câu |
| `03_ragas_scores.png` | Bảng so sánh RAGAS V1 vs V2 trên terminal |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json` |
| `04_pii_demo_log.txt` | Demo PIIDetector (6 case) |
| `04_json_demo_log.txt` | Demo JSONFormatter (5 case) |

## Trace công khai trên LangSmith

- Bước 1 (`rag-query`): https://smith.langchain.com/public/d526efb5-14fb-4d5f-802b-258df4859951/r
- Bước 2 (`ab-rag-query`): https://smith.langchain.com/public/d71758bd-b2ea-45d1-832e-3eb46e17115d/r

Mỗi trace hiển thị câu hỏi, 3 đoạn context do retriever trả về, prompt, lời gọi LLM và câu trả lời.

## Cấu hình

- LLM: OpenAI `gpt-4o-mini`, embeddings `text-embedding-3-small` (cũng dùng làm evaluator của RAGAS, `temperature=0`)
- Chunking: `RecursiveCharacterTextSplitter` 500 ký tự / overlap 50 → 107 chunks, FAISS, retriever `k=3`
- A/B routing: `md5(request_id) % 2` → V1 = 19 câu, V2 = 31 câu (cùng `request_id` luôn ra cùng phiên bản)

## Hai phiên bản prompt

| | V1 — ngắn gọn | V2 — chuyên gia, có cấu trúc |
|---|---|---|
| Độ dài | 2-4 câu | 1 câu tóm tắt + 3-5 gạch đầu dòng |
| Cách diễn đạt | Ưu tiên dùng lại câu chữ của context | Tổng hợp, sắp xếp các facts thành ý chính |
| Ràng buộc | Chỉ dùng context, không biết thì nói không biết | Mỗi ý phải được context hỗ trợ, nói rõ phần còn thiếu |

## Kết quả RAGAS (50 cặp QA × 2 phiên bản)

| Metric | V1 | V2 | Tốt hơn |
|---|---|---|---|
| faithfulness | **0.9803** | 0.8619 | V1 |
| answer_relevancy | **0.8986** | 0.8825 | V1 |
| context_recall | 1.0000 | 1.0000 | Hòa |
| context_precision | **0.9483** | 0.9417 | V1 (chênh không đáng kể) |

Mục tiêu faithfulness ≥ 0.8 đạt ở cả 2 phiên bản; V1 đạt ≥ 0.9.

## Phân tích: vì sao V1 tốt hơn V2

1. **Faithfulness (chênh lớn nhất, 0.98 vs 0.86).** RAGAS tách câu trả lời thành các claim rồi kiểm tra từng claim có suy ra được từ context hay không. V1 bắt trả lời 2-4 câu và dùng lại câu chữ của context, nên số claim ít và gần như trích nguyên văn, gần như claim nào cũng được hỗ trợ. V2 buộc phải viết **3-5 gạch đầu dòng** kể cả khi context chỉ có 1-2 fact liên quan. Để lấp đủ số ý, model diễn giải, khái quát hoặc thêm hệ quả "hợp lý" (vd. giải thích thêm tác dụng của một kỹ thuật), và những claim này không có trong 3 chunk được retrieve nên bị tính là không trung thực. Nói cách khác, ràng buộc về **định dạng/độ dài** của V2 mâu thuẫn với ràng buộc "chỉ dùng context".

2. **Answer relevancy (0.90 vs 0.88).** Metric này sinh ngược câu hỏi từ câu trả lời rồi so embedding với câu hỏi gốc. Câu trả lời V1 ngắn, tập trung đúng vào câu hỏi; câu trả lời V2 có thêm câu tóm tắt và các ý phụ nên câu hỏi sinh ngược "loãng" hơn, điểm giảm nhẹ.

   Ngoài ra, trong `02_ab_routing_log.txt` (câu 25, prompt V2) model trả lời **bằng tiếng Việt** cho câu hỏi tiếng Anh, dù prompt yêu cầu "trả lời bằng cùng ngôn ngữ với câu hỏi". System prompt tiếng Việt dài hơn của V2 kéo model sang tiếng Việt; câu trả lời khác ngôn ngữ với câu hỏi và context cũng làm giảm answer_relevancy và faithfulness.

3. **Context recall / precision gần như bằng nhau.** Hai metric này đánh giá **retriever** (context so với reference và câu hỏi), không phụ thuộc câu trả lời. Cả hai phiên bản dùng chung FAISS index và `k=3`, nên context giống hệt nhau: recall = 1.0 ở cả hai, precision chỉ chênh 0.007 do LLM-judge không hoàn toàn tất định.

**Kết luận:** với RAG trên knowledge base ngắn, prompt ngắn gọn bám sát context (V1) cho câu trả lời trung thực và liên quan hơn. Nếu muốn giữ format có cấu trúc của V2, nên bỏ yêu cầu số lượng cố định ("3-5 gạch đầu dòng") và thay bằng "tối đa N ý, chỉ liệt kê ý có trong context" để không ép model thêm thông tin.

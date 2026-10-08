"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "nguyen-ha-rag-prompt-v1"
PROMPT_V2_NAME = "nguyen-ha-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — ngắn gọn: trả lời trực tiếp 2-4 câu, bám sát câu chữ trong context.
SYSTEM_V1 = (
    "Bạn là trợ lý hỏi đáp ngắn gọn. Trả lời câu hỏi trong 2-4 câu, "
    "chỉ sử dụng thông tin có trong context bên dưới và ưu tiên dùng lại cách diễn đạt của context. "
    "Không thêm kiến thức bên ngoài. Nếu context không chứa câu trả lời, hãy nói rằng bạn không biết. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2 — có cấu trúc: giọng chuyên gia, mở đầu bằng 1 câu tóm tắt rồi liệt kê
# các facts chính dưới dạng gạch đầu dòng (3-5 ý), mỗi ý phải lấy từ context.
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích tài liệu kỹ thuật. Đọc kỹ context, xác định các facts liên quan "
    "trực tiếp đến câu hỏi, rồi trình bày câu trả lời có cấu trúc: "
    "một câu tóm tắt ở đầu, sau đó 3-5 gạch đầu dòng nêu các ý chính. "
    "Mỗi ý phải được hỗ trợ bởi context; không suy đoán hay bổ sung thông tin ngoài context. "
    "Nếu context không đủ thông tin, hãy nói rõ phần nào còn thiếu. "
    "Trả lời bằng cùng ngôn ngữ với câu hỏi.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Nếu prompt không đổi, Hub trả về 409 "Nothing to commit" — không phải lỗi.
    """
    for tag, name, template, desc in [
        ("V1", PROMPT_V1_NAME, PROMPT_V1, "V1 – ngắn gọn, 2-4 câu"),
        ("V2", PROMPT_V2_NAME, PROMPT_V2, "V2 – chuyên gia, có cấu trúc"),
    ]:
        try:
            url = client.push_prompt(name, object=template, description=desc)
            print(f"Đã push {tag}: {url}")
        except Exception as e:
            if "Nothing to commit" in str(e):
                print(f"{tag} không thay đổi, Hub giữ nguyên phiên bản mới nhất của '{name}'")
            else:
                print(f"{tag} lỗi khi push: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"Dùng local fallback cho '{PROMPT_V1_NAME}' ({e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"Dùng local fallback cho '{PROMPT_V2_NAME}' ({e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic),
    khác với random hay hash() built-in của Python (bị salt mỗi lần chạy).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:55]}...")
        print(f"     A: {result['answer'][:100]!r}")

    print(f"\nRouting: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()

"""Reproducible evaluation harness for InvoiceAI's retrieval and answers.

Example:
    python evaluation.py path/to/invoice.pdf --cases eval_cases.json
    python evaluation.py --make-template eval_cases.json

You must fill the case file using values verified from the PDF. This script
never invents expected invoice values and never counts unanswerable questions
as retrieval hits.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CASES = [
    {
        "id": "invoice_number",
        "question": "What is the invoice number?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Copy the exact invoice number from the PDF. Add valid alternative formats to accepted_answers if needed.",
    },
    {
        "id": "vendor",
        "question": "Who is the vendor?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Use the vendor name exactly as it appears in the PDF.",
    },
    {
        "id": "total_amount",
        "question": "What is the total amount due?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "number",
        "notes": "Use the actual amount. Include alternate decimal/currency display forms if appropriate.",
    },
    {
        "id": "due_date",
        "question": "What is the due date?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "If the PDF does not contain a due date, set answerable=false and accepted_answers=[] after verifying the whole document.",
    },
    {
        "id": "client",
        "question": "Who is the client or bill-to party?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Use the exact client name shown in the document.",
    },
    {
        "id": "invoice_date",
        "question": "What is the invoice date?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Use the date format shown on the invoice, plus any equivalent accepted date formatting.",
    },
    {
        "id": "tax_amount",
        "question": "What tax amount is shown on the invoice?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "number",
        "notes": "If there is no tax field, mark answerable=false after checking the entire PDF.",
    },
    {
        "id": "payment_terms",
        "question": "What are the payment terms?",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Add a concise exact phrase or acceptable equivalent from the invoice.",
    },
    {
        "id": "line_items",
        "question": "List the main line items on the invoice.",
        "answerable": True,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "Enter a short exact item/product phrase that appears in the invoice. If line items are absent, mark answerable=false after reviewing all pages.",
    },
    {
        "id": "unanswerable_control",
        "question": "What is the capital of France?",
        "answerable": False,
        "accepted_answers": [],
        "evidence_terms": [],
        "match_mode": "text",
        "notes": "A known out-of-document control. Correct answer behavior is abstention, not a factual answer.",
    },
]

ABSTENTION_PATTERNS = (
    "not found in document",
    "not present in the document",
    "not available in the document",
    "cannot find that information in the document",
    "does not contain that information",
    "not specified in the document",
)


def normalize_text(value: str) -> str:
    """Lowercase and normalize punctuation/whitespace for robust text matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _phrase_found(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def _numeric_values(text: str) -> set[str]:
    """Extract normalized numeric values, allowing commas and currency symbols."""
    numbers = re.findall(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?", str(text))
    result: set[str] = set()
    for raw in numbers:
        try:
            number = float(raw.replace(",", ""))
            result.add(f"{number:.6f}".rstrip("0").rstrip("."))
        except ValueError:
            continue
    return result


def is_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(pattern) in normalized for pattern in ABSTENTION_PATTERNS)


def _answer_only(answer: str) -> str:
    """Exclude a quoted source snippet so citations cannot make a wrong answer pass."""
    text = str(answer).split("\nSource:", 1)[0].split("\nsource:", 1)[0].strip()
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    return text


def answer_matches(answer: str, case: dict[str, Any]) -> bool:
    """Score the final answer (not its source quote) against verified answers."""
    answer_text = _answer_only(answer)
    if case.get("answerable", True) is False:
        return is_abstention(answer_text)
    accepted = case.get("accepted_answers") or []
    if not accepted:
        raise ValueError(f"Case '{case.get('id', '?')}' has no accepted_answers.")
    if case.get("match_mode", "text") == "number":
        answer_numbers = _numeric_values(answer_text)
        for expected in accepted:
            if _numeric_values(str(expected)) & answer_numbers:
                return True
        return False
    return any(_phrase_found(answer_text, str(expected)) for expected in accepted)


def retrieval_matches(retrieved: list[dict[str, Any]], case: dict[str, Any]) -> bool:
    """Check whether annotated evidence appears in retrieved chunks.

    This is an evidence-term hit metric—not a semantic human relevance score.
    Unanswerable cases are deliberately excluded from retrieval-hit rate.
    """
    if case.get("answerable", True) is False:
        raise ValueError("retrieval_matches should only be used for answerable cases.")
    evidence_terms = case.get("evidence_terms") or []
    if not evidence_terms:
        raise ValueError(f"Case '{case.get('id', '?')}' has no evidence_terms.")
    combined = "\n".join(item["doc"].page_content for item in retrieved)
    return any(_phrase_found(combined, str(term)) for term in evidence_terms)


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("The cases file must contain a non-empty 'test_cases' array.")
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id", "").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError("Every test case needs a unique, non-empty id.")
        seen_ids.add(case_id)
        if not case.get("question", "").strip():
            raise ValueError(f"Case '{case_id}' needs a question.")
        if case.get("answerable", True):
            if not case.get("accepted_answers") or not case.get("evidence_terms"):
                raise ValueError(
                    f"Case '{case_id}' is answerable but lacks accepted_answers or evidence_terms. "
                    "Fill them with values verified from the PDF."
                )
            labels = [str(value) for value in list(case.get("accepted_answers", [])) + list(case.get("evidence_terms", []))]
            if any(re.search(r"COPY_REAL|FILL_IN|TODO|PLACEHOLDER", value, flags=re.IGNORECASE) for value in labels):
                raise ValueError(f"Case '{case_id}' still contains a placeholder. Replace it with verified PDF ground truth.")
        if case.get("match_mode", "text") not in {"text", "number"}:
            raise ValueError(f"Case '{case_id}' has unsupported match_mode; use 'text' or 'number'.")


def create_template(path: Path) -> None:
    payload = {
        "instructions": (
            "Edit each answerable case using the exact PDF as ground truth. "
            "If a requested field is genuinely absent after checking all pages, mark answerable=false. "
            "Do not leave placeholder/empty answers for answerable cases."
        ),
        "test_cases": DEFAULT_CASES,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Created template: {path}\nFill accepted_answers and evidence_terms from your actual PDF before evaluating.")


def run_evaluation(pdf_path: Path, cases_path: Path, output_path: Path, k: int = 4) -> dict[str, Any]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("test_cases", payload) if isinstance(payload, dict) else payload
    validate_cases(cases)

    # Imports are delayed so this file's scoring functions can be tested without
    # installing every LLM dependency.
    from rag.ingestion import build_vector_db, load_pdf, split_documents
    from rag.retrieval import answer_query, retrieve_chunks

    print(f"Loading and indexing: {pdf_path}")
    docs = load_pdf(str(pdf_path))
    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError("No text chunks were extracted from the PDF; evaluation cannot run.")
    db = build_vector_db(chunks)

    results: list[dict[str, Any]] = []
    retrieval_hits = 0
    answer_correct = 0
    answerable_count = 0
    abstention_correct = 0
    unanswerable_count = 0
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        error = None
        retrieved: list[dict[str, Any]] = []
        answer = ""
        try:
            retrieved = retrieve_chunks(db, case["question"], k=k)
            # Reuse this exact retrieved result: don't perform a second retrieval.
            response = answer_query(db, case["question"], k=k, retrieved=retrieved)
            answer = response.get("answer", "")
            if case.get("answerable", True):
                retrieval_hit = retrieval_matches(retrieved, case)
                retrieval_hits += int(retrieval_hit)
                answerable_count += 1
            else:
                retrieval_hit = None  # not part of retrieval-hit denominator
                unanswerable_count += 1
                abstention_correct += int(is_abstention(answer))
            correct = answer_matches(answer, case)
            answer_correct += int(correct)
        except Exception as exc:
            retrieval_hit = False if case.get("answerable", True) else None
            correct = False
            error = f"{type(exc).__name__}: {exc}"
            if case.get("answerable", True):
                answerable_count += 1
            else:
                unanswerable_count += 1
        latency = round(time.perf_counter() - started, 3)
        latencies.append(latency)
        result_row = {
            "id": case["id"],
            "question": case["question"],
            "answerable": bool(case.get("answerable", True)),
            "answer": answer,
            "retrieval_hit": retrieval_hit,
            "answer_correct": correct,
            "latency_sec": latency,
            "error": error,
            "retrieved_sources": [
                {
                    "source_file": item["doc"].metadata.get("source_file", "document"),
                    "page": item["doc"].metadata.get("page_display", "?"),
                    "distance": item.get("distance"),
                }
                for item in retrieved[:4]
            ],
        }
        results.append(result_row)
        retrieval_label = "n/a" if retrieval_hit is None else ("HIT" if retrieval_hit else "MISS")
        print(f"{'PASS' if correct else 'FAIL'} answer | retrieval {retrieval_label} | {latency:.3f}s | {case['id']}")
        if error:
            print(f"  Error: {error}")
        print(f"  Answer: {answer[:180]}\n")

    total = len(cases)
    answer_rate = round(answer_correct / total * 100, 2) if total else 0.0
    retrieval_rate = round(retrieval_hits / answerable_count * 100, 2) if answerable_count else None
    abstention_rate = round(abstention_correct / unanswerable_count * 100, 2) if unanswerable_count else None
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document": pdf_path.name,
        "cases_file": cases_path.name,
        "k": k,
        "definition": {
            "retrieval_hit_rate": "Correctly annotated evidence term appears in at least one retrieved chunk; answerable cases only.",
            "answer_accuracy": "Generated answer matches a manually verified accepted answer, or abstains on a verified unanswerable case.",
            "abstention_accuracy": "Unanswerable cases for which the system gives a recognized abstention response.",
            "latency": "Wall-clock time for retrieval plus answer generation; includes external API/network latency.",
        },
        "counts": {
            "total_cases": total,
            "answerable_cases": answerable_count,
            "unanswerable_cases": unanswerable_count,
            "retrieval_hits": retrieval_hits,
            "correct_answers": answer_correct,
            "correct_abstentions": abstention_correct,
        },
        "retrieval_hit_rate_percent": retrieval_rate,
        "answer_accuracy_percent": answer_rate,
        "abstention_accuracy_percent": abstention_rate,
        "average_latency_sec": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_sec": round(statistics.median(latencies), 3) if latencies else None,
        "results": results,
        "limitations": [
            "Scores depend on the correctness and coverage of the manually annotated test cases.",
            "Evidence-term matching is a transparent proxy for retrieval success, not a full semantic relevance judgment.",
            "Results from a small test set should not be generalized to all invoices.",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 64)
    print(f"Answer accuracy:       {answer_correct}/{total} ({answer_rate}%)")
    if retrieval_rate is None:
        print("Retrieval hit rate:    N/A (no answerable cases)")
    else:
        print(f"Retrieval hit rate:    {retrieval_hits}/{answerable_count} ({retrieval_rate}%) [answerable cases only]")
    if abstention_rate is not None:
        print(f"Abstention accuracy:   {abstention_correct}/{unanswerable_count} ({abstention_rate}%)")
    print(f"Average latency:       {report['average_latency_sec']} sec/question")
    print(f"Detailed report saved: {output_path}")
    print("=" * 64)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate InvoiceAI against manually labeled PDF ground truth.")
    parser.add_argument("pdf", nargs="?", help="Path to the invoice PDF used for evaluation")
    parser.add_argument("--cases", default="eval_cases.json", help="JSON file containing verified test cases")
    parser.add_argument("--output", default="eval_report.json", help="Path for the detailed JSON report")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (1-10)")
    parser.add_argument("--make-template", metavar="PATH", help="Create a labeled-cases template and exit")
    args = parser.parse_args()

    if args.make_template:
        create_template(Path(args.make_template))
        return 0
    if not args.pdf:
        parser.error("provide a PDF path, or use --make-template eval_cases.json")
    if not Path(args.pdf).is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if not Path(args.cases).is_file():
        parser.error(f"Case file not found: {args.cases}. Create one with --make-template eval_cases.json")
    if not 1 <= args.k <= 10:
        parser.error("--k must be between 1 and 10")
    try:
        run_evaluation(Path(args.pdf), Path(args.cases), Path(args.output), args.k)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

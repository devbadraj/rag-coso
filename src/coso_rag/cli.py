import argparse
import json
import sys

from .config import DOCUMENTS
from .openrouter import APIError, OpenRouter


def main():
    parser = argparse.ArgumentParser(
        description="Ask the supplied construction documents, with citations"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser(
        "ingest", help="Extract pages and rebuild the local index; cached OCR is free"
    )
    ingest.add_argument(
        "--vision",
        action="store_true",
        help="Allow paid extraction for uncached scan pages",
    )
    sub.add_parser("train", help="Fine-tune the multilingual text ranker locally")
    sub.add_parser(
        "evaluate",
        help="Run baseline and trained retrieval evaluation without paid calls",
    )
    live = sub.add_parser(
        "evaluate-live", help="Explicit paid answer evaluation on test questions"
    )
    live.add_argument("--limit", type=int, default=None)
    cost = sub.add_parser("cost", help="Show this app's spending and reservations")
    cost.add_argument(
        "--history", action="store_true", help="Include individual request records"
    )
    ask = sub.add_parser("ask", help="Ask a question or retrieve evidence only")
    ask.add_argument("question")
    ask.add_argument("--retrieve-only", action="store_true", help="No answer API call")
    mode = ask.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument(
        "--trained",
        action="store_true",
        help="Use the fine-tuned text ranker",
    )
    ask.add_argument("--document", choices=[document["id"] for document in DOCUMENTS])
    ask.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            from .ingest import extract
            from .retrieval import build_index

            print(json.dumps(build_index(extract(vision=args.vision)), indent=2))
        elif args.command == "train":
            from .training import train

            print(json.dumps(train(), indent=2))
        elif args.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate()
            print(
                json.dumps(
                    {
                        **result,
                        "splits": {
                            s: {
                                mode: {k: v for k, v in m.items() if k != "details"}
                                for mode, m in v.items()
                            }
                            for s, v in result["splits"].items()
                        },
                    },
                    indent=2,
                )
            )
        elif args.command == "cost":
            usage = OpenRouter().usage()
            if not args.history:
                usage.pop("history")
            print(json.dumps(usage, indent=2))
        elif args.command == "evaluate-live":
            from .live_evaluation import evaluate_live

            print(json.dumps(evaluate_live(args.limit), indent=2))
        elif args.command == "ask":
            from .answering import answer, format_answer, gather_evidence
            from .retrieval import Retriever

            r = Retriever()
            trained = True if args.trained else False if args.baseline else None
            if args.retrieve_only:
                evidence = gather_evidence(
                    args.question, r, trained=trained, document=args.document
                )
                if args.json:
                    print(json.dumps(evidence, ensure_ascii=False, indent=2))
                else:
                    print("Retrieved evidence (no generated answer):")
                    for c in evidence:
                        print(
                            f"\n[{c['id']}] {c['title']}, PDF page {c['page']}\n{c['text']}"
                        )
            else:
                value = answer(
                    args.question, r, trained=trained, document=args.document
                )
                print(
                    json.dumps(value, ensure_ascii=False, indent=2)
                    if args.json
                    else format_answer(value)
                )
                if not args.json:
                    cost = (
                        f"${value['cost_usd']:.6f}"
                        if value["cost_usd"] is not None
                        else "unreported (reserved)"
                    )
                    print(f"\nReported API cost: {cost}; cached: {value['cached']}")
    except (APIError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

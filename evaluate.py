"""Run the 5 evaluation questions from planning.md end-to-end and print results.

    python evaluate.py
"""
from guide.generate import generate

QUESTIONS = [
    "How hard is COP 3514 (Program Design) and what should I focus on to do well?",
    "Which professor is best for CDA 3201 (Computer Logic & Design)?",
    "Is Dr. Small a good professor for COP 2510, and are her exams open book?",
    "What are the easiest CS electives at USF?",
    "What programming languages does the USF CS program teach and in what order?",
]


def main() -> None:
    for i, q in enumerate(QUESTIONS, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        out = generate(q)
        print(out["answer"])
        print(f"\n[backend: {out['backend']}]  Retrieved sources:")
        for s in out["sources"]:
            print(f"  [{s['n']}] dist={s['distance']}  {s['citation']}")
        print()


if __name__ == "__main__":
    main()

"""Command line interface for the statistical translation aligner."""

from __future__ import annotations

import argparse

from statistical_translation.corpus import SentencePair
from statistical_translation.data import gold_pairs, train_pairs
from statistical_translation.evaluation import alignment_error_rate
from statistical_translation.ibm1 import IBMModel1
from statistical_translation.ibm2 import IBMModel2


def _cmd_evaluate(args: argparse.Namespace) -> None:
    pairs = train_pairs()
    gold = gold_pairs()
    model1 = IBMModel1()
    model1.train(pairs, iterations=args.iterations)
    model2 = IBMModel2()
    model2.train(pairs, iterations=args.iterations)
    report1 = alignment_error_rate(model1, gold)
    report2 = alignment_error_rate(model2, gold)
    print("Alignment error rate on held-out gold alignments (lower is better):")
    print(f"  Model 1: {report1.format()}")
    print(f"  Model 2: {report2.format()}")


def _cmd_align(args: argparse.Namespace) -> None:
    pairs = train_pairs()
    model = IBMModel2()
    model.train(pairs, iterations=args.iterations)
    pair = SentencePair.from_text(args.source, args.target)
    source = pair.source_with_null()
    print(f"Alignment of '{args.target}' to '{args.source}':")
    for target_index, source_index in model.align(pair).pairs:
        source_word = source[source_index] if source_index > 0 else "(null)"
        print(f"  {pair.target[target_index]:<12} -> {source_word}")


def _cmd_translate(args: argparse.Namespace) -> None:
    model = IBMModel1()
    model.train(train_pairs(), iterations=args.iterations)
    print(f"Most probable translations of '{args.word}':")
    for word, probability in model.translate_word(args.word, k=args.k):
        print(f"  {word:<12} {probability:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="statistical-translation", description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="alignment error rate for Model 1 and Model 2")

    align = sub.add_parser("align", help="align a source and target sentence")
    align.add_argument("source")
    align.add_argument("target")

    translate = sub.add_parser("translate", help="most probable translations of a source word")
    translate.add_argument("word")
    translate.add_argument("--k", type=int, default=3)

    handlers = {"evaluate": _cmd_evaluate, "align": _cmd_align, "translate": _cmd_translate}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()

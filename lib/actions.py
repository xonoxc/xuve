from argparse import Namespace, ArgumentParser
from lib.keyword_search import (
    bm25search,
    calc_bm25_idf,
    calc_bm25_tf,
    search,
    term_freq,
    inverse_document_freq,
    tf_idf,
)
from lib.semantic_search import validate_model


COMMANDS = {
    "search": {
        "intro": lambda a: f"Searching For: {a.query} ....",
        "action": lambda a: search(a.query),
        "format": lambda r: (
            "No results found."
            if not r
            else "\n".join([f"{i + 1}. {res['title']}" for i, res in enumerate(r)])
        ),
    },
    "tf": {
        "intro": lambda a: f"Finding term frequency for: {a.term} ....",
        "action": lambda a: term_freq(
            int(a.doc_id),
            a.term,
        ),
        "format": lambda r, a: (
            f"Term Frequency of '{a.term}' in Document ID {a.doc_id}: {r}"
        ),
    },
    "tfidf": {
        "intro": lambda a: f"Finding tfidf for: {a.term} ....",
        "action": lambda a: tf_idf(
            int(a.doc_id),
            a.term,
        ),
        "format": lambda r, a: (
            f"TF-IDF score of '{a.term}' in document '{a.doc_id}': {r:.2f}"
        ),
    },
    "idf": {
        "intro": lambda a: f"Calculating the idf value for term {a.term} ....",
        "action": lambda a: inverse_document_freq(a.term),
        "format": lambda r, a: (f"Inverse document frequency of '{a.term}': {r:.2f}"),
    },
    "bm25idf": {
        "intro": lambda a: f"Calculating the bm25_idf value for term {a.term} ....",
        "action": lambda a: calc_bm25_idf(a.term),
        "format": lambda r, a: (
            f"BM25 Inverse document frequency of '{a.term}': {r:.2f}"
        ),
    },
    "bm25tf": {
        "intro": lambda a: f"Calculating the bm25_tf value for term {a.term} ....",
        "action": lambda a: calc_bm25_tf(
            int(a.doc_id),
            a.term,
            K1=float(a.k1),
            b=float(a.b),
        ),
        "format": lambda r,
        a: f"BM25 TF score of '{a.term}' in document '{a.doc_id}': {r:.2f}",
    },
    "bm25search": {
        "intro": lambda a: f"searching results for term {a.query} ....",
        "action": lambda a: bm25search(a.query, a.limit),
        "format": lambda results: "\n".join(
            f"{i + 1}. ({doc_id}) {title} - Score: {score:.2f}"
            for i, (doc_id, title, score) in enumerate(results)
        ),
    },
    "verify": {
        "intro": "verifying model.....",
        "action": lambda: validate_model(),
        "format": lambda: print("\n"),
    },
}


def act(args: Namespace, parser: ArgumentParser) -> None:
    cmd = COMMANDS.get(args.command)
    if not cmd:
        parser.print_help()
        return

    # intro can be a callable or string
    intro = cmd["intro"](args) if callable(cmd["intro"]) else cmd["intro"]
    print(intro)

    # action may or may not take args
    try:
        result = cmd["action"](args)
    except TypeError:
        result = cmd["action"]()

    # format may or may not exist or take args
    if not cmd.get("format"):
        return

    try:
        output = cmd["format"](result, args)
    except TypeError:
        try:
            output = cmd["format"](result)
        except TypeError:
            output = cmd["format"]()

    if output:
        print(output)

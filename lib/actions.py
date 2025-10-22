from argparse import Namespace, ArgumentParser
from lib.keyword_search import (
    calc_bm25_idf,
    calc_bm25_tf,
    search,
    term_freq,
    inverse_document_freq,
    tf_idf,
)


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
}


def act(args: Namespace, parser: ArgumentParser) -> None:
    cmd = COMMANDS.get(args.command)
    if not cmd:
        parser.print_help()
        return

    print(cmd["intro"](args))
    result = cmd["action"](args)

    # some commands need args in format
    try:
        print(
            cmd["format"](result, args),
        )
    except TypeError:
        print(cmd["format"](result))

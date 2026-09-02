"""Insert the minted code-archive DOI everywhere it belongs, then rebuild both documents.

The DOI of the code archive cannot exist before the archive is deposited, so the
manuscript and the response letter carry a marked placeholder until then.  Once
the deposit is made (see ``archive/README.md``), run::

    python set_code_doi.py 10.5281/zenodo.1234567

This replaces the placeholder in all five places a reader sees it -- twice in the
manuscript (Data availability, Code availability) and three times in the response
letter (E.2, R2.10, R4.7) -- and rebuilds paper.pdf and
response_to_reviewers_submission.pdf so the typeset documents agree with the source.

Pass ``--check`` to report where the placeholder still stands without changing
anything.  The script refuses a DOI that does not look like one, and (unless
``--no-resolve`` is given) refuses one that does not resolve at doi.org, because a
DOI printed in a submitted manuscript must lead the reader to the real archive.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#: The literal placeholder each file carries, and how many times it must occur.
TARGETS: Tuple[Tuple[str, str, int], ...] = (
    ("paper.tex", "[DOI TO BE INSERTED]", 2),
    ("response_to_reviewers_submission.tex",
     "[repository DOI --- to be inserted on deposit]", 1),   # a \doifield macro, used 3x
    ("response_to_reviewers_submission.txt",
     "[repository DOI - to be inserted on deposit]", 3),
)

#: A DOI is "10." then a registrant prefix, a slash, and a non-empty suffix.
DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$")


def read(path: str) -> str:
    with io.open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def write(path: str, text: str) -> None:
    with io.open(os.path.join(PROJECT_ROOT, path), "w", encoding="utf-8",
                 newline="") as handle:
        handle.write(text)


def resolves(doi: str, timeout: float = 20.0) -> Tuple[bool, str]:
    """True when doi.org knows this DOI (HEAD via the content-negotiation API)."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8", "replace")
        return ('"responseCode":1' in payload.replace(" ", "")), "doi.org: found"
    except urllib.error.HTTPError as exc:
        return False, "doi.org: HTTP %s" % exc.code
    except Exception as exc:                       # network down, DNS, timeout
        return False, "doi.org: unreachable (%s)" % exc


def survey() -> List[Tuple[str, str, int, int]]:
    """(path, placeholder, expected, found) for every target file."""
    rows = []
    for path, placeholder, expected in TARGETS:
        if not os.path.exists(os.path.join(PROJECT_ROOT, path)):
            rows.append((path, placeholder, expected, -1))
            continue
        rows.append((path, placeholder, expected, read(path).count(placeholder)))
    return rows


def build(doi: str) -> List[str]:
    """Rebuild both PDFs; returns the log lines worth showing."""
    notes = []
    for stem, passes in (("paper", 3), ("response_to_reviewers_submission", 3)):
        if not os.path.exists(os.path.join(PROJECT_ROOT, stem + ".tex")):
            continue
        steps = ["pdflatex"]
        if stem == "paper":
            steps = ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
        else:
            steps = ["pdflatex"] * passes
        for step in steps:
            command = [step, "-interaction=nonstopmode", stem] if step == "pdflatex" \
                else [step, stem]
            result = subprocess.run(command, cwd=PROJECT_ROOT,
                                    capture_output=True, text=True)
            if step == "bibtex" and result.returncode != 0:
                notes.append("%s: bibtex exit %d" % (stem, result.returncode))
        log = os.path.join(PROJECT_ROOT, stem + ".log")
        if os.path.exists(log):
            with io.open(log, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            pages = re.search(r"Output written on .*?\((\d+) pages", text)
            errors = len(re.findall(r"^! ", text, re.MULTILINE))
            notes.append("%-34s %s pages, %d TeX errors, %d overfull"
                         % (stem + ".pdf", pages.group(1) if pages else "?",
                            errors, text.count("Overfull")))
            if doi not in text and stem == "paper":
                pass          # the log need not contain it; the source check covers us
    return notes


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("doi", nargs="?", help="the minted DOI, e.g. 10.5281/zenodo.1234567")
    parser.add_argument("--check", action="store_true",
                        help="report where the placeholder still stands; change nothing")
    parser.add_argument("--no-resolve", action="store_true",
                        help="skip the doi.org resolution check (offline use)")
    parser.add_argument("--no-build", action="store_true", help="do not rebuild the PDFs")
    args = parser.parse_args(argv)

    rows = survey()
    print("Placeholder survey:")
    problems = []
    for path, _placeholder, expected, found in rows:
        state = "MISSING FILE" if found < 0 else ("%d of %d" % (found, expected))
        print("  %-42s %s" % (path, state))
        if found != expected:
            problems.append(path)

    if args.check:
        print("\nnothing changed (--check)")
        return 1 if problems else 0

    if not args.doi:
        parser.error("a DOI is required unless --check is given")

    doi = args.doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):].strip()
    if not DOI_RE.match(doi):
        print("\nERROR: %r does not look like a DOI (expected 10.NNNN/suffix)" % doi)
        return 2

    if not args.no_resolve:
        ok, note = resolves(doi)
        print("\nResolution check: %s" % note)
        if not ok:
            print("ERROR: %s does not resolve. A DOI printed in a submitted "
                  "manuscript must lead the reader to the real archive.\n"
                  "       Deposit first, then run this with the minted DOI "
                  "(or pass --no-resolve if you are offline and certain)." % doi)
            return 3

    if problems:
        print("\nERROR: the placeholder is not where it is expected in: %s\n"
              "       Has it already been replaced? Run with --check."
              % ", ".join(problems))
        return 4

    print("\nInserting %s" % doi)
    for path, placeholder, expected, _found in rows:
        text = read(path)
        write(path, text.replace(placeholder, doi))
        print("  %-42s %d occurrence(s)" % (path, expected))

    if not args.no_build:
        print("\nRebuilding:")
        for note in build(doi):
            print("  " + note)

    left = [p for p, _ph, _e, f in survey() if f]
    print("\nDone. Remaining placeholder occurrences: %d"
          % sum(f for _p, _ph, _e, f in survey() if f > 0))
    if left:
        print("WARNING: still present in %s" % ", ".join(left))
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())

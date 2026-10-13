#!/usr/bin/env python3

"""Explain "could not find term" rows in a metadata exceptions report.

make_ingest_sheet.py checks every controlled value against the local taxonomy
cache (Utility_Files/taxonomies.csv). When a value isn't found it logs
"could not find term in <vocabulary> taxonomy" but doesn't say why. This
script reads that report and the cache, then sorts each missing term into the
most likely cause so you know what to fix.

Categories:
    unknown vocabulary   The vocabulary name isn't in the cache at all, so
                         every term checked against it is flagged. This points
                         to a mapping file or cache problem, not a data one.
    different spelling   The same term exists in this vocabulary with
                         different capitalization or spacing.
    alternate name       The value matches an alternate name of a term. The
                         script only accepts the preferred name.
    other vocabulary     The term exists, but in a different vocabulary.
    not in cache         No match anywhere. Either the term is new (create it
                         in Drupal) or the cache is out of date (refresh it).

Usage:
    python3 check_taxonomy_misses.py \\
        logs/2026-10-13-143439_metadata_exceptions.csv

    python3 check_taxonomy_misses.py <exceptions.csv> \\
        --cache Utility_Files/taxonomies.csv \\
        --cache-metadata Utility_Files/taxonomies_cache_metadata.json \\
        --output misses.csv
"""

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

# Standard library imports
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = SCRIPT_DIR / 'Utility_Files' / 'taxonomies.csv'
DEFAULT_CACHE_METADATA = (
    SCRIPT_DIR / 'Utility_Files' / 'taxonomies_cache_metadata.json'
)

MISS_PATTERN = re.compile(
    r'^could not find term in (?P<vocabulary>.+) taxonomy$'
)

CATEGORY_ORDER = [
    'unknown vocabulary',
    'different spelling',
    'alternate name',
    'other vocabulary',
    'not in cache',
]

NEXT_STEPS = {
    'unknown vocabulary': (
        "Check the taxonomy name in the mapping CSV against the cache's "
        "Vocabulary column, then refresh the cache."
    ),
    'different spelling': (
        "Correct the spelling or capitalization in the sheet to match the "
        "existing term."
    ),
    'alternate name': (
        "Use the term's preferred name in the sheet."
    ),
    'other vocabulary': (
        "Check that the column maps to the intended vocabulary."
    ),
    'not in cache': (
        "Refresh the cache first. If it is still missing, create the term "
        "in Drupal."
    ),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Explain 'could not find term' rows in a metadata exceptions "
            "report."
        )
    )
    parser.add_argument(
        'exceptions_csv',
        type=Path,
        help="Path to a *_metadata_exceptions.csv report.",
    )
    parser.add_argument(
        '--cache',
        type=Path,
        default=DEFAULT_CACHE,
        help="Taxonomy cache CSV (default: Utility_Files/taxonomies.csv).",
    )
    parser.add_argument(
        '--cache-metadata',
        type=Path,
        default=DEFAULT_CACHE_METADATA,
        help="Cache metadata JSON, used to report how old the cache is.",
    )
    parser.add_argument(
        '--output',
        type=Path,
        help=(
            "Where to write the per-term results "
            "(default: next to the exceptions report)."
        ),
    )

    return parser.parse_args()


def normalize(text: str) -> str:
    """Collapse whitespace and ignore case so spellings can be compared."""
    return ' '.join(str(text).split()).casefold()


def describe_cache_age(metadata_path: Path) -> str:
    """Describe when the taxonomy cache was last refreshed.

    Args:
        metadata_path: Path to the cache metadata JSON.

    Returns:
        A sentence describing the cache's age, or why it is unknown.
    """
    if not metadata_path.exists():
        return f"No cache metadata found at {metadata_path}."

    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        refreshed = datetime.fromisoformat(metadata['last_refreshed'])
    except (ValueError, KeyError, OSError):
        return f"Could not read the refresh date from {metadata_path}."

    age = datetime.now(refreshed.tzinfo) - refreshed
    days = age.days

    return (
        f"The taxonomy cache was last refreshed "
        f"{refreshed.strftime('%Y-%m-%d %H:%M')} "
        f"({days} day{'s' if days != 1 else ''} ago)."
    )


def build_indexes(cache: pd.DataFrame) -> dict:
    """Index the taxonomy cache for quick matching.

    Args:
        cache: Taxonomy cache with Vocabulary, Name, and Alternate Names.

    Returns:
        Dictionary of lookups keyed by vocabulary or normalized name.
    """
    vocabularies = set(cache['Vocabulary'])
    names_by_vocab = {}
    alternates_by_vocab = {}
    vocabs_by_name = {}

    for vocabulary, name, alternates in zip(
        cache['Vocabulary'],
        cache['Name'],
        cache.get('Alternate Names', [''] * len(cache)),
    ):
        key = normalize(name)
        names_by_vocab.setdefault(vocabulary, {}).setdefault(key, name)
        vocabs_by_name.setdefault(key, set()).add(vocabulary)

        for alternate in re.split(r'[|;]', str(alternates)):
            if alternate.strip():
                alternates_by_vocab.setdefault(vocabulary, {})[
                    normalize(alternate)
                ] = name

    return {
        'vocabularies': vocabularies,
        'names_by_vocab': names_by_vocab,
        'alternates_by_vocab': alternates_by_vocab,
        'vocabs_by_name': vocabs_by_name,
    }


def classify(vocabulary: str, term: str, indexes: dict) -> tuple[str, str]:
    """Work out the most likely reason a term wasn't found.

    Args:
        vocabulary: Vocabulary the term was checked against.
        term: The value that wasn't found.
        indexes: Lookups from build_indexes.

    Returns:
        Tuple of category and a short detail.
    """
    if vocabulary not in indexes['vocabularies']:
        return 'unknown vocabulary', f"'{vocabulary}' is not in the cache"

    key = normalize(term)
    existing = indexes['names_by_vocab'].get(vocabulary, {}).get(key)

    if existing is not None:
        return 'different spelling', f"existing term is '{existing}'"

    preferred = indexes['alternates_by_vocab'].get(vocabulary, {}).get(key)

    if preferred is not None:
        return 'alternate name', f"preferred name is '{preferred}'"

    elsewhere = sorted(indexes['vocabs_by_name'].get(key, set()))

    if elsewhere:
        return 'other vocabulary', f"found in {', '.join(elsewhere)}"

    return 'not in cache', ''


def main() -> None:
    """Report why terms were not found."""
    args = parse_arguments()

    for path in (args.exceptions_csv, args.cache):
        if not path.exists():
            sys.exit(f"File not found: {path}")

    exceptions = pd.read_csv(
        args.exceptions_csv,
        dtype=str,
        keep_default_na=False,
    )
    cache = pd.read_csv(args.cache, dtype=str, keep_default_na=False)

    missing = {'Vocabulary', 'Name'} - set(cache.columns)

    if missing:
        sys.exit(f"Cache is missing column(s): {', '.join(sorted(missing))}")

    print(describe_cache_age(args.cache_metadata))

    # Pick out the term-not-found rows and the vocabulary each names
    matches = exceptions['exception'].str.extract(MISS_PATTERN)
    rows = exceptions.loc[matches['vocabulary'].notna()].copy()
    rows['vocabulary'] = matches.loc[rows.index, 'vocabulary']

    print(
        f"{len(rows)} 'could not find term' row(s) out of "
        f"{len(exceptions)} in {args.exceptions_csv.name}."
    )

    if rows.empty:
        return

    indexes = build_indexes(cache)
    summary = (
        rows.groupby(['vocabulary', 'value'])
        .size()
        .reset_index(name='occurrences')
    )
    classified = summary.apply(
        lambda row: classify(row['vocabulary'], row['value'], indexes),
        axis=1,
        result_type='expand',
    )
    summary[['category', 'detail']] = classified
    summary['next_step'] = summary['category'].map(NEXT_STEPS)
    summary['category'] = pd.Categorical(
        summary['category'],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(
        ['category', 'vocabulary', 'value']
    ).reset_index(drop=True)

    print(f"\n{len(summary)} distinct term(s) by likely cause:\n")

    for category in CATEGORY_ORDER:
        group = summary[summary['category'] == category]

        if group.empty:
            continue

        print(f"  {category}: {len(group)}")
        print(f"    {NEXT_STEPS[category]}")

        for row in group.head(10).itertuples():
            detail = f" ({row.detail})" if row.detail else ''
            print(
                f"      [{row.vocabulary}] {row.value}"
                f" x{row.occurrences}{detail}"
            )

        if len(group) > 10:
            print(f"      ... and {len(group) - 10} more")

    output = args.output or args.exceptions_csv.with_name(
        f"{args.exceptions_csv.stem}_taxonomy_misses.csv"
    )
    summary.to_csv(output, index=False)
    print(f"\nFull list written to {output}")


if __name__ == '__main__':
    main()


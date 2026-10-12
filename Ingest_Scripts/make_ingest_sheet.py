#!/usr/bin/env python3

"""Generate Workbench ingest sheets from Islandora Workbench exports and metadata sheets.

This script combines Islandora Workbench exports and descriptive metadata from 
Google Sheets or local spreadsheet files, validates and transforms the metadata 
into Islandora-compatible fields, and generates one or more Workbench ingest
spreadsheets. It supports create, update, and publish workflows while
performing schema validation, controlled vocabulary checking, EDTF date
validation, and detailed logging of metadata issues and transformations.

Usage:
    # Create ingest sheets from Google Sheets
    python3 make_ingest_sheet.py \
        --user_id abc123 \
        --ingest_task create \
        --batch_path /workbench/batches/example \
        --metadata_id <metadata_sheet_id> \
        --id-column identifier

    # Read a specific tab of a Google Sheet (the first tab is the default)
    python3 make_ingest_sheet.py \
        --user_id abc123 \
        --ingest_task create \
        --batch_path /workbench/batches/example \
        --metadata_id <metadata_sheet_id> \
        --metadata_tab "Batch 5" \
        --id-column identifier

    # Update existing objects using local spreadsheets
    python3 make_ingest_sheet.py \
        --ingest_task update \
        --batch_path /workbench/batches/example \
        --export_sheet /workbench/batches/example/export/example.csv \
        --metadata_sheet /workbench/batches/example/metadata.csv \
        --id-column identifier

    # Publish existing objects
    python3 make_ingest_sheet.py \
        --ingest_task update \
        --metadata_level publish \
        --publish y \
        --batch_path /workbench/batches/example \
        --metadata_id <metadata_sheet_id> \
        --id-column identifier

    # Run interactively
    python3 make_ingest_sheet.py
"""

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

# Standard library imports
import argparse
import logging
import re
import os
import sys
import traceback
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Third-party imports
import pandas as pd
import requests
from edtf import parse_edtf
from tqdm import tqdm

# Local imports
from definitions import (
    COMMA_DELIMITED_FIELDS,
    CONTROLLED_FIELDS,
    DATE_FIELDS,
    DISALLOWED_FIELDS_BY_INGEST_TASK,
    DOMAINS,
    FIELDS,
    FORMATTED_FIELDS,
    GOOGLE_CREDENTIALS_FILE,
    LINKED_AGENT_TYPES,
    MANDATORY_FIELDS,
    MANIFEST_FIELD_MAPPING,
    MARC_FIELD_MAPPING,
    MINIMAL_METADATA_FIELDS,
    MODEL_MAPPING,
    PUBLISH_FIELDS,
    RELATOR_TERMS,
    SCAN_BATCH_DIR_FIELDS,
    TEMPLATE_FIELD_MAPPING,
    UTILITY_FILES_DIR
)
from process_mods import check_if_agent_field
from taxonomy_manager import (
    TAXONOMIES,
    load_taxonomies,
)
from utilities import (
    DRUPAL_EXTENDED_EDTF_PATTERN,
    ERROR_SYMBOL,
    SUCCESS_SYMBOL,
    WARNING_SYMBOL,
    LogRegistry,
    cap_first,
    create_df,
    create_directory,
    df_to_csv,
    normalize_for_join,
    prompt_for_input,
    read_google_sheet,
    remove_whitespaces,
    setup_logger,
    write_reports,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 10000
LOGGER_NAME = LogRegistry.MAKE_INGEST_SHEET

# Object models that are always top-level; any parent_id on these is ignored
TOP_LEVEL_MODELS = ('Paged Content', 'Compound Object', 'Newspaper')

# Columns holding files for Workbench's additional_files setting. Each one
# needs a matching commented entry in default_create_config.yml, such as
# "# - transcript: 9", where the number is the Media Use term ID.
ADDITIONAL_FILE_COLUMNS = ('transcript', 'thumbnail', 'hocr', 'extracted_text')

# Columns whose values are file paths or URLs for Workbench to fetch. They
# are not Drupal field values, so the 255-character text limit doesn't apply.
FILE_REFERENCE_COLUMNS = ('file',) + ADDITIONAL_FILE_COLUMNS


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Application configuration values."""

    user_id: str
    batch_path: str | Path
    batch_size: int
    export_id: str | None
    metadata_id: str | None
    id_column: str
    credentials_file: str
    ingest_task: str
    metadata_level: str
    publish: bool
    remove_pages: bool
    export_sheet: str | None = None
    metadata_sheet: str | None = None
    metadata_tab: str | None = None
    refresh_taxonomies: bool = False
    batch_dir: str | None = None
    timestamp: str | None = None
    file_prefix: str | None = None
    output_dir: Path | None = None
    log_dir: Path | None = None
    log_path: Path | None = None

    def log_configuration(
        self,
        logger: logging.Logger,
    ) -> None:
        """Log the current application configuration.

        Only values that are not None are included.
        """
        logger.info("Configuration Input:")

        for field in fields(self):
            value = getattr(self, field.name)

            if value in (None, ""):
                continue

            logger.info(
                "  %-18s %s",
                field.name,
                value,
            )


@dataclass
class ProcessingResult:
    """Runtime processing state and report data."""
    
    issues: list[dict] = field(default_factory=list)
    transformations: list[dict] = field(default_factory=list)
    current_batch: int = 1
    unexpected_failure_count: int = 0

    def log_issue(
        self,
        pid: str,
        field: str | None,
        value: str | list[str] | None,
        issue: str,
        log_msg: str | None = None,
    ) -> None:
        """Record a metadata issue and write it to the logger."""
        self.issues.append({
            'batch': self.current_batch,
            'pid': pid,
            'field': field,
            'value': value,
            'exception': issue,
        })

        if not log_msg:
            log_msg = (
                f"{cap_first(issue)} in field '{field}' with value '{value}'."
            )

        logging.getLogger(LOGGER_NAME).error(
            "Record %s: %s",
            pid,
            log_msg
        )

    def log_transformation(
        self,
        pid: str,
        field: str,
        old_value: str,
        new_value: str,
        transformation: str,
        log_msg: str | None = None,
    ) -> None:
        """Record a metadata transformation and write it to the logger."""
        self.transformations.append({
            'batch': self.current_batch,
            'pid': pid,
            'field': field,
            'old_value': old_value,
            'new_value': new_value,
            'transformation': transformation,
        })

        if not log_msg:
            log_msg = (
                f"{cap_first(transformation)} in field '{field}': "
                f"'{old_value}' -> '{new_value}'."
            )

        logging.getLogger(LOGGER_NAME).info(
            "Record %s: %s",
            pid,
            log_msg
        )

    def increment_failure_count(self) -> None:
        """Increment the count of records that failed unexpectedly."""
        self.unexpected_failure_count += 1

    def advance_batch(self) -> None:
        """Advance the current batch counter."""
        self.current_batch += 1


class PrefixedValue(str):
    """String that remembers the vocabulary prefix and bare term it holds.
 
    Behaves exactly like the combined string (``prefix + term``) for joining,
    comparison, and output, but keeps the two parts separate so that
    de-duplication never has to guess where a prefix ends. Parsing the
    combined string by splitting on ``:`` is unreliable because the bare term
    itself may contain colons (e.g. ``Sound recording: musical``).
    """
 
    def __new__(cls, term: str, prefix: str | None = None):
        prefix = prefix or ''
        obj = super().__new__(cls, f'{prefix}{term}')
        obj.term = term
        obj.prefix = prefix
        return obj


@dataclass(frozen=True)
class FieldMapping:
    """Islandora field mapping information for a source CSV field."""

    field: str | None = None
    taxonomy: str | None = None
    prefix: str | None = None
    repeatable: bool = False

    @property
    def is_mapped(self) -> bool:
        """Return whether a mapping was found."""
        return self.field is not None


@dataclass(frozen=True)
class MappingResolution:
    """Outcome of resolving a source CSV field against the mapping tables.

    Resolution depends only on the CSV field name, so it is computed once per
    field and cached for the run. Anything worth logging is stored here
    rather than logged during the lookup; it is logged per record, and only
    for records that actually have data in the field.
    """

    mapping: FieldMapping = field(default_factory=FieldMapping)

    # Linked-agent lookup details (role-based columns such as
    # "photographer_person")
    agent_role_term: str | None = None
    agent_issue: str | None = None
    relator_code: str | None = None
    relator_note: str | None = None

    # How an unmatched field was ultimately handled: "i2_field" when the
    # name is itself an I2 field, "unmapped" when it is dropped
    fallback: str | None = None


@dataclass
class FamilyOrder:
    """Ingest sheet rows arranged so parents precede their children.

    All per-row lists and dictionaries use positions in the reordered
    ``sheet``, not the original row order.
    """

    # Ingest sheet in processing order
    sheet: pd.DataFrame

    # True for the last row of each family; batches may only end here
    family_end: list[bool] = field(default_factory=list)

    # Cleaned record identifier for each row
    keys: list[str] = field(default_factory=list)

    # Cleaned parent identifier for each row ('' for top-level rows)
    parents: list[str] = field(default_factory=list)

    # Rows whose parent_id matches no record in the sheet
    unresolved_parents: set[int] = field(default_factory=set)

    # Rows caught in a circular parent_id reference
    circular: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class TaxonomyLookup:
    """Taxonomy terms indexed by vocabulary for fast validation."""

    # Vocabulary -> every term name in it
    names: dict[str, frozenset[str]]

    # Vocabulary -> every term ID in it
    term_ids: dict[str, frozenset[str]]

    # Vocabulary -> term ID -> name of the first term with that ID
    id_to_name: dict[str, dict[str, str]]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# --- Argument Parsing and Input Loading ---

def parse_arguments() -> AppConfig:
    """Parse command-line arguments and prompt for missing required values.

    Returns:
        Application configuration object.
    """
    parser = argparse.ArgumentParser(
        description="Process CSV files in batches."
    )

    parser.add_argument(
        '-u',
        '--user_id',
        type=str,
        help="The user ID to associate with the operation.",
    )
    parser.add_argument(
        '-t',
        '--ingest_task',
        type=str,
        choices=['create', 'update'],
        help="Workbench task: 'create' or 'update'.",
    )
    parser.add_argument(
        '-b',
        '--batch_path',
        type=str,
        help="Path to a batch directory for Workbench ingests.",
    )
    parser.add_argument(
        '-z',
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of records per batch. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        '-e',
        '--export_sheet',
        type=str,
        help="Path to export sheet on local device.",
    )
    parser.add_argument(
        '--export_id',
        type=str,
        help="Google Sheet ID for the export file.",
    )
    parser.add_argument(
        '-d',
        '--metadata_id',
        type=str,
        help="Google Sheet ID for the metadata file.",
    )
    parser.add_argument(
        '--metadata_tab',
        type=str,
        help=(
            "Tab name within the metadata Google Sheet. Defaults to the "
            "first tab. Only valid with --metadata_id."
        ),
    )
    parser.add_argument(
        '--metadata_sheet',
        type=str,
        help="Path to metadata sheet on local device.",
    )
    parser.add_argument(
        '--id-column',
        type=str,
        help="Metadata sheet column that contains the record identifier.",
    )
    parser.add_argument(
        '-c',
        '--credentials_file',
        type=str,
        default=GOOGLE_CREDENTIALS_FILE,
        help="Path to the Google service account credentials JSON.",
    )
    parser.add_argument(
        '-l',
        '--metadata_level',
        type=str,
        choices=['minimal', 'complete', 'publish'],
        help="Metadata detail level: 'minimal', 'complete', or 'publish'.",
    )
    parser.add_argument(
        '-p',
        '--publish',
        type=str,
        choices=['y', 'n'],
        help="Specify whether the ingest batch should be published.",
    )
    parser.add_argument(
        '--remove_pages',
        action='store_true',
        help="Remove objects with a page model from ingest sheet.",
    )
    parser.add_argument(
        '--refresh_taxonomies',
        action='store_true',
        help="Refresh the taxonomy cache before processing records.",
    )
    args = parser.parse_args()

    if not args.user_id:
        args.user_id = prompt_for_input("Enter your Pitt user ID: ")

    if not args.ingest_task:
        args.ingest_task = prompt_for_input(
            "Enter the Workbench ingest task (create/update): ",
            valid_choices=['create', 'update'],
        )

    if not args.batch_path:
        args.batch_path = prompt_for_input(
            "Enter the path to the Workbench batch directory: "
        )

    if not args.metadata_id and not args.metadata_sheet:
        args.metadata_id = prompt_for_input(
            "Enter the Google Sheet ID for the metadata sheet: "
        )

    if not args.id_column:
        args.id_column = prompt_for_input(
            "Enter the metadata sheet column name used as the identifier: "
        )

    if not args.metadata_level:
        args.metadata_level = prompt_for_input(
            "Enter the metadata level (minimal, complete, or publish): ",
            valid_choices=['minimal', 'complete', 'publish'],
        )

    if not args.publish:
        args.publish = prompt_for_input(
            "Should the ingest batch be published (y/n)?: ",
            valid_choices=['y', 'n'],
        )

    if args.batch_size < 1:
        raise ValueError("--batch_size must be a positive integer.")

    if args.export_id and args.export_sheet:
        raise ValueError(
            "Provide either --export_id or --export_sheet, not both."
        )

    if args.metadata_id and args.metadata_sheet:
        raise ValueError(
            "Provide either --metadata_id or --metadata_sheet, not both."
        )

    if args.metadata_tab and not args.metadata_id:
        raise ValueError(
            "--metadata_tab can only be used with --metadata_id."
        )

    args.publish = args.publish == 'y'

    return AppConfig(**vars(args))


def load_input_sheets(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load export and metadata sheets.

    Args:
        config: Application configuration object.

    Returns:
        Tuple containing export and metadata DataFrames.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if config.export_id:
        logger.info(
            "Using export sheet from Google ID: %s",
            config.export_id,
        )
        export_df = read_google_sheet(
            sheet_id=config.export_id,
            credentials_file=config.credentials_file,
        )
    elif config.export_sheet:
        logger.info(
            "Using export sheet from local file: %s",
            config.export_sheet,
        )
        export_df = create_df(config.export_sheet)
    else:
        logger.info("No export sheet provided.")
        export_df = pd.DataFrame()

    if config.metadata_id:
        logger.info(
            "Using metadata sheet from Google ID: %s (tab: %s)",
            config.metadata_id,
            config.metadata_tab or 'first tab',
        )
        metadata_df = read_google_sheet(
            config.metadata_id,
            sheet_name=config.metadata_tab,
            credentials_file=config.credentials_file,
        )
    elif config.metadata_sheet:
        logger.info(
            "Using metadata sheet from local file: %s",
            config.metadata_sheet,
        )
        metadata_df = create_df(config.metadata_sheet)
    else:
        logger.info("No metadata sheet provided.")
        metadata_df = pd.DataFrame()

    return export_df, metadata_df


# --- Sheet Merging ---

def validate_id_column(metadata_df: pd.DataFrame, id_column: str) -> None:
    """Confirm that the metadata sheet contains the identifier column.

    Args:
        metadata_df: Human-editable metadata sheet.
        id_column: Name of the column that holds the record identifier.

    Raises:
        ValueError: If the column is not in the metadata sheet.
    """
    if id_column not in metadata_df.columns:
        available = ', '.join(map(str, metadata_df.columns))
        message = (
            f"Identifier column '{id_column}' was not found in the metadata "
            f"sheet. Available columns: {available}"
        )
        logging.getLogger(LOGGER_NAME).error(message)
        raise ValueError(message)


def merge_sheets(
    export_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Workbench node IDs to a metadata sheet.

    The metadata sheet is treated as the authoritative source. The export
    sheet contributes only the system-generated ``node_id``, matched using the
    export's ``id`` column and the metadata sheet's ``id_column``. The metadata
    sheet's original identifier column name is preserved.

    Args:
        export_df: Workbench export containing ``id`` and ``node_id``.
        metadata_df: Human-editable metadata sheet.
        id_column: Metadata sheet column that holds the record identifier.

    Returns:
        Tuple containing:
            - the metadata sheet with matched ``node_id`` values added; and
            - metadata rows that could not be matched to the export.

    Raises:
        KeyError: If the export is missing ``id`` or ``node_id``.
        ValueError: If the metadata sheet is missing ``id_column``, either
            input contains blank normalized identifiers, or either input
            contains duplicate normalized identifiers.
    """
    logger = logging.getLogger(LOGGER_NAME)

    export_df = export_df.copy()
    metadata_df = metadata_df.copy()

    # Ensure all required columns exist in the export sheet
    required_export_columns = {
        'id',
        'node_id',
    }
    missing_export_columns = (
        required_export_columns - set(export_df.columns)
    )

    if missing_export_columns:
        columns = ', '.join(sorted(missing_export_columns))
        message = (
            "Export sheet is missing required column(s): "
            f"{columns}"
        )
        logger.error(message)
        raise KeyError(message)

    # Retain only the export fields needed for the merge
    export_df = export_df[
        ['id', 'node_id']
    ].copy()

    # Confirm the metadata identifier field exists
    validate_id_column(metadata_df, id_column)
    metadata_id_field = id_column

    logger.info(
        "Using '%s' as the metadata join field.",
        metadata_id_field,
    )

    # Create normalized join keys
    export_df['__export_id_join__'] = normalize_for_join(
        export_df['id']
    )
    metadata_df['__metadata_id_join__'] = normalize_for_join(
        metadata_df[metadata_id_field]
    )

    # Reject identifiers that are null or become blank after normalization
    export_blank_ids = (
        export_df['__export_id_join__'].isna()
        | export_df['__export_id_join__']
        .astype('string')
        .str.strip()
        .eq('')
    )

    if export_blank_ids.any():
        blank_count = int(export_blank_ids.sum())
        row_numbers = (
            export_df.index[export_blank_ids]
            .to_series()
            .add(2)
            .astype(str)
            .tolist()
        )
        examples = ', '.join(row_numbers[:10])

        message = (
            f"Export sheet contains {blank_count} blank normalized "
            f"identifier(s) in 'id'. Example spreadsheet row(s): "
            f"{examples}."
        )
        logger.error(message)
        raise ValueError(message)

    metadata_blank_ids = (
        metadata_df['__metadata_id_join__'].isna()
        | metadata_df['__metadata_id_join__']
        .astype('string')
        .str.strip()
        .eq('')
    )

    if metadata_blank_ids.any():
        blank_count = int(metadata_blank_ids.sum())
        row_numbers = (
            metadata_df.index[metadata_blank_ids]
            .to_series()
            .add(2)
            .astype(str)
            .tolist()
        )
        examples = ', '.join(row_numbers[:10])

        message = (
            f"Metadata sheet contains {blank_count} blank normalized "
            f"identifier(s) in '{metadata_id_field}'. Example spreadsheet "
            f"row(s): {examples}."
        )
        logger.error(message)
        raise ValueError(message)

    # Check for duplicate normalized IDs in the export sheet
    export_duplicates = (
        export_df['__export_id_join__']
        .value_counts()
    )
    export_duplicates = export_duplicates[
        export_duplicates > 1
    ]

    if not export_duplicates.empty:
        examples = ', '.join(
            map(str, export_duplicates.index[:10])
        )
        message = (
            "Export sheet contains duplicate normalized IDs. "
            f"Examples: {examples}"
        )
        logger.error(message)
        raise ValueError(message)

    # Check for duplicate normalized identifiers in the metadata sheet
    metadata_duplicates = (
        metadata_df['__metadata_id_join__']
        .value_counts()
    )
    metadata_duplicates = metadata_duplicates[
        metadata_duplicates > 1
    ]

    if not metadata_duplicates.empty:
        examples = ', '.join(
            map(str, metadata_duplicates.index[:10])
        )
        message = (
            "Metadata sheet contains duplicate normalized identifiers. "
            f"Examples: {examples}"
        )
        logger.error(message)
        raise ValueError(message)

    # Remove node_id column from metadata DataFrame, since the Workbench export 
    # sheet is authoritative for node IDs. If a stale or manually-entered
    # node_id happened to exist in the metadata sheet, dropping it here
    # guarantees the merge below always uses the export's current value
    # instead of silently keeping outdated data.
    metadata_df.drop(
        columns=['node_id'],
        errors='ignore',
        inplace=True,
    )

    # Add the export's node_id to the corresponding metadata row
    ingest_sheet = metadata_df.merge(
        export_df[
            ['__export_id_join__', 'node_id']
        ],
        how='left',
        left_on='__metadata_id_join__',
        right_on='__export_id_join__',
        validate='one_to_one',
    )

    matched_count = int(
        ingest_sheet['__export_id_join__'].notna().sum()
    )
    logger.info(
        "Matched %d of %d metadata record(s) to the export.",
        matched_count,
        len(ingest_sheet),
    )

    # Identify metadata rows whose identifiers did not match the export
    unmatched_mask = (
        ingest_sheet['__export_id_join__'].isna()
    )
    unmatched = ingest_sheet.loc[
        unmatched_mask
    ].copy()

    if not unmatched.empty:
        logger.warning(
            "%d metadata row(s) could not be matched to the export.",
            len(unmatched),
        )

    # Clean up helper columns and export sheet 'id' column
    helper_columns = [
        '__export_id_join__',
        '__metadata_id_join__',
    ]

    ingest_sheet.drop(
        columns=helper_columns,
        errors='ignore',
        inplace=True,
    )
    unmatched.drop(
        columns=helper_columns,
        errors='ignore',
        inplace=True,
    )

    return ingest_sheet, unmatched


# --- Batch Helpers ---

def order_records_by_family(
    ingest_sheet: pd.DataFrame,
    id_column: str,
    use_hierarchy: bool = True,
) -> FamilyOrder:
    """Arrange ingest sheet rows so every parent precedes its children.

    Workbench can only resolve a ``parent_id`` that refers to an ``id``
    earlier in the same CSV, so each top-level record is followed directly
    by all of its descendants. Siblings are ordered by ``field_weight``
    when any of them has one, with unweighted siblings after the weighted
    ones in sheet order. When none of the siblings has a ``field_weight``,
    they are ordered by ``node_id`` instead, then by sheet order.

    A record whose ``parent_id`` matches nothing in the sheet (for example,
    a child of an object that already exists in Drupal) is treated as the
    start of its own family and flagged in the result. Records caught in a
    circular reference are flagged and placed last, in sheet order.

    Sheets without a ``parent_id`` column are returned unchanged, as are
    all sheets when ``use_hierarchy`` is False.

    Args:
        ingest_sheet: Ingest sheet in original row order.
        id_column: Column that holds the record identifier.
        use_hierarchy: Whether parent_id should be used. When False, rows
            keep their original order and are treated as unrelated.

    Returns:
        FamilyOrder with the reordered sheet and per-row family details.
    """
    row_count = len(ingest_sheet)

    if (
        row_count == 0
        or not use_hierarchy
        or 'parent_id' not in ingest_sheet.columns
    ):
        return FamilyOrder(
            sheet=ingest_sheet,
            family_end=[True] * row_count,
            keys=[''] * row_count,
            parents=[''] * row_count,
        )

    def cleaned_column(column: str) -> list[str]:
        if column not in ingest_sheet.columns:
            return [''] * row_count

        return [
            '' if pd.isna(value) else remove_whitespaces(str(value))
            for value in ingest_sheet[column]
        ]

    def numeric_column(column: str) -> list[float]:
        if column not in ingest_sheet.columns:
            return [float('nan')] * row_count

        return pd.to_numeric(
            ingest_sheet[column],
            errors='coerce',
        ).tolist()

    keys = cleaned_column(id_column)
    models = cleaned_column('field_model')
    weights = numeric_column('field_weight')
    node_ids = numeric_column('node_id')

    # parent_id is repeatable in the schema, but a record has one parent
    parents = []

    for value in cleaned_column('parent_id'):
        tokens = split_and_clean(value)
        parents.append(tokens[0] if tokens else '')

    # Top-level models never have a parent, whatever the sheet says
    parents = [
        '' if models[i] in TOP_LEVEL_MODELS else parent
        for i, parent in enumerate(parents)
    ]

    first_position = {}

    for position, key in enumerate(keys):
        if key and key not in first_position:
            first_position[key] = position

    children = {}

    for position, parent in enumerate(parents):
        if parent:
            children.setdefault(parent, []).append(position)

    def order_siblings(positions: list[int]) -> list[int]:
        """Order a group of siblings by weight, or else by node ID."""
        sort_values = (
            weights
            if any(not pd.isna(weights[p]) for p in positions)
            else node_ids
        )

        return sorted(
            positions,
            key=lambda p: (
                pd.isna(sort_values[p]),
                0 if pd.isna(sort_values[p]) else sort_values[p],
                p,
            ),
        )

    visited = set()

    def collect_family(start: int) -> list[int]:
        """Return a record followed by all of its descendants."""
        family = []
        stack = [start]

        while stack:
            position = stack.pop()

            if position in visited:
                continue

            visited.add(position)
            family.append(position)

            if keys[position]:
                siblings = order_siblings(children.get(keys[position], []))
                stack.extend(reversed(siblings))

        return family

    families = []
    unresolved = set()
    emitted_orphan_groups = set()

    for position in range(row_count):
        parent = parents[position]

        if not parent:
            if position not in visited:
                families.append(collect_family(position))

        elif parent not in first_position:
            # Parent is not in this sheet. Records that share a missing
            # parent are siblings, so order them together, at the point
            # where the first of them appears.
            if parent in emitted_orphan_groups:
                continue

            emitted_orphan_groups.add(parent)

            for member in order_siblings(children[parent]):
                unresolved.add(member)

                if member not in visited:
                    families.append(collect_family(member))

    # Anything left is part of, or hangs from, a circular reference
    circular = {
        position for position in range(row_count) if position not in visited
    }

    for position in sorted(circular):
        families.append([position])

    order = [position for family in families for position in family]
    family_end = []

    for family in families:
        family_end.extend([False] * (len(family) - 1) + [True])

    new_position = {original: new for new, original in enumerate(order)}

    return FamilyOrder(
        sheet=ingest_sheet.iloc[order],
        family_end=family_end,
        keys=[keys[original] for original in order],
        parents=[parents[original] for original in order],
        unresolved_parents={new_position[p] for p in unresolved},
        circular={new_position[p] for p in circular},
    )


def should_flush_batch(buffer: list, batch_size: int) -> bool:
    """Determine whether the current batch is large enough to flush.

    This is a minimum, not a hard limit. Callers only flush at the end of
    a family of records, so a batch can exceed ``batch_size`` rather than
    split a parent from its children.

    Args:
        buffer: Processed record buffer.
        batch_size: Number of records at which a batch may be flushed.

    Returns:
        True if batch should be flushed; otherwise False.
    """
    return len(buffer) >= batch_size


def flush_batch(
    buffer: list,
    batch_count: int,
    config: AppConfig,
) -> pd.DataFrame:
    """Write the current buffer to CSV and prepare a config file.

    Args:
        buffer: Processed records.
        batch_count: Current batch number.
        config: Application configuration object.

    Returns:
        DataFrame written from the record buffer.
    """
    sub_batch_prefix = (
        f'{config.file_prefix}_{batch_count}_'
        f'{config.ingest_task}_{config.metadata_level}'
    )
    sub_batch_file = f'{sub_batch_prefix}.csv'
    sub_batch_path = config.output_dir / sub_batch_file

    records_df = save_records(buffer, sub_batch_path, config)

    if records_df is None:
        return pd.DataFrame()

    # Check for additional media files
    media_files = [
        column for column in ADDITIONAL_FILE_COLUMNS
        if column in records_df.columns
    ]

    # Prepare config file
    prepare_config(
        sub_batch_prefix,
        sub_batch_file,
        config.batch_path,
        config.batch_dir,
        config.user_id,
        config.ingest_task,
        media_files,
    )

    return records_df


def save_records(
    records: list,
    destination: str | Path,
    config: AppConfig,
) -> pd.DataFrame | None:
    """Convert records to a DataFrame, filter fields, and write to CSV.

    Args:
        records: Record dictionaries to write.
        destination: Output CSV path.
        config: Application configuration object.

    Returns:
        DataFrame written to CSV, or None if no records are provided.
    """
    if not records:
        print("No records to save. The output file will not be created.")
        return None

    # Convert list of dictionaries to DataFrame
    records_df = pd.DataFrame.from_dict(records)

    # Filter to keep only fields for minimal metadata ingest
    if config.metadata_level == 'minimal':
        records_df = filter_minimal_metadata_fields(records_df)

    # Filter fields to those allowable by ingest task
    records_df = filter_fields(records_df, config.ingest_task)

    # Ensure that parent_id is empty for top-level object models. Records
    # arrive already ordered with parents before children (see
    # order_records_by_family), so they are not re-sorted here.
    if (
        'parent_id' in records_df.columns
        and 'field_model' in records_df.columns
    ):
        records_df.loc[
            records_df['field_model'].isin(TOP_LEVEL_MODELS),
            'parent_id',
        ] = pd.NA

    # Write the resulting DataFrame to a CSV file
    df_to_csv(records_df, destination)

    # Report creation of processed CSV path
    formatted_path = Path(destination).as_posix()
    print(f"\nIngest file saved: {formatted_path}")

    return records_df


def setup_batch_directory(batch_path: Path) -> None:
    """Set up the batch directory structure.

    Creates the root batch directory and all required processing
    subdirectories if they do not already exist.

    Args:
        batch_path: Path to the batch directory.
    """
    # Create subdirectories
    batch_subdirs = [
        'configs',
        'export',
        'import',
        'logs',
        'metadata',
        'rollback',
        'tmp',
    ]

    for subdir in batch_subdirs:
        create_directory(batch_path / subdir)


def prepare_config(
    batch_prefix: str,
    batch_file: str,
    batch_path: Path,
    batch_dir: str,
    user_id: str,
    ingest_task: str,
    media_files: list[str]
) -> Path | None:
    """Read and customize the import config file for a specific batch.

    Args:
        batch_prefix: Pattern used for naming output batch files.
        batch_file: CSV filename for output batch file.
        batch_path: Full path to the root of the batch folder.
        batch_dir: Name of the batch directory (used in config substitutions).
        user_id: User ID to insert into the config.
        ingest_task: Ingest task to insert into the config ('create' or
            'update').
        media_files: I2 fields for additional media files.

    Returns:
        Path to the customized config file, or None if the default config is
        missing.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # The project's .env file is loaded when definitions.py is imported
    import_password = os.getenv('ISLANDORA_PASSWORD', '')

    # Define source and destination config paths. Each ingest task has its
    # own template, e.g. default_create_config.yml or default_update_config.yml
    default_config = f'default_{ingest_task}_config.yml'
    config_src = UTILITY_FILES_DIR / default_config
    config_filename = f'{batch_prefix}.yml'
    config_dest = batch_path / 'configs' / config_filename

    # Ensure configs directory exists
    create_directory(config_dest.parent)

    # Read and update placeholders from default config
    if not config_src.exists():
        message = (
            f"{default_config} not found in Utility_Files; no config file "
            "was written for this batch."
        )
        logger.error(message)
        print(message)
        return None

    with config_src.open('r', encoding='utf-8') as f:
        content = f.read()

    if '[ISLANDORA_PASSWORD]' in content and not import_password:
        logger.warning(
            "ISLANDORA_PASSWORD is not set, so the password in %s will be "
            "blank.",
            config_dest.name,
        )

    # Replace placeholders
    content = content.replace('[ISLANDORA_PASSWORD]', import_password)
    content = content.replace('[BATCH_DIRECTORY]', batch_dir)
    content = content.replace('[BATCH_PREFIX]', batch_prefix)
    content = content.replace('[BATCH_FILE]', batch_file)
    content = content.replace('[USER_ID]', user_id)
    content = content.replace('[INGEST_TASK]', ingest_task)

    # Uncomment the additional media file types used in this batch
    enabled_media_files = []

    for column in ADDITIONAL_FILE_COLUMNS:
        if column not in media_files:
            continue

        comment = f'# - {column}'

        if comment not in content:
            logger.warning(
                "%s has no '%s' entry, so the '%s' column will not be "
                "configured as an additional file.",
                default_config,
                comment,
                column,
            )
            continue

        content = content.replace(comment, comment[2:])
        enabled_media_files.append(column)

    if enabled_media_files:
        content = content.replace('#additional_files', 'additional_files')

    # Write the customized config to the destination
    with config_dest.open('w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nConfig file saved: {config_dest.as_posix()}")
    return config_dest


# --- Record Initialization and Formatting ---

def initialize_record() -> dict:
    """Initialize a record with all output fields as empty lists.

    Returns:
        Record dictionary initialized with expected fields.
    """
    return {
        field: []
        for field in FIELDS.Field
    }


def split_and_clean(text: str, extra_delimiters: str = '') -> list[str]:
    """Tokenize a string by pipe and semi-colon delimiters.

    Args:
        text: Raw input string.
        extra_delimiters: Additional single-character delimiters to split
            on, beyond the default pipe and semicolon (e.g. ',' for fields
            populated via a comma-joined multi-select dropdown).

    Returns:
        Cleaned list of non-empty values.
    """
    if not text:
        return []

    # Split on pipe or semicolon, or any extra delimiters, plus any surrounding 
    # whitespace
    delimiter_chars = '|;' + extra_delimiters
    parts = re.split(rf'\s*[{re.escape(delimiter_chars)}]\s*', text)

    # Clean up individual parts and filter out empty strings
    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def format_record(record: dict) -> dict:
    """Serialize record list values and remove empty fields.

    Args:
        record: Metadata record.

    Returns:
        Cleaned record with list values converted to pipe-delimited strings.
    """
    if not record:
        return record

    for field, values in list(record.items()):
        if not isinstance(values, list):
            continue

        if values:
            record[field] = '|'.join(values)
        else:
            del record[field]

    return record


def filter_fields(
    records: pd.DataFrame,
    ingest_task: str,
) -> pd.DataFrame:
    """Drop fields that are not allowed for the specified ingest task.

    Args:
        records: DataFrame containing ingest records.
        ingest_task: Ingest task type, either 'create' or 'update'.

    Returns:
        DataFrame with disallowed fields removed.

    Raises:
        ValueError: If ingest_task is invalid.
    """
    if ingest_task not in DISALLOWED_FIELDS_BY_INGEST_TASK:
        raise ValueError(
            f"Invalid ingest_task: {ingest_task!r}. "
            "Expected 'create' or 'update'."
        )

    disallowed_fields = set(DISALLOWED_FIELDS_BY_INGEST_TASK[ingest_task])

    # Update tasks change node fields only. Files are added or replaced with
    # other Workbench tasks (e.g., add_media and update_media), and Workbench
    # rejects CSV columns that aren't Drupal fields, so file columns never
    # belong in an update CSV.
    if ingest_task == 'update':
        dropped_files = [
            column for column in FILE_REFERENCE_COLUMNS
            if column in records.columns
        ]

        if dropped_files:
            logging.getLogger(LOGGER_NAME).info(
                "Removed file column(s) from the update ingest sheet: %s.",
                ', '.join(dropped_files),
            )

        disallowed_fields |= set(FILE_REFERENCE_COLUMNS)

    fields_to_drop = [
        field for field in disallowed_fields
        if field in records.columns
    ]

    return records.drop(columns=fields_to_drop).copy()


def filter_minimal_metadata_fields(records_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only minimal metadata fields in an output DataFrame.
    
        Args:
            records_df: Records to filter.
    
        Returns:
            Records containing only publish fields.

    """
    fields_to_keep = [
        field for field in records_df.columns
        if field in MINIMAL_METADATA_FIELDS
    ]

    return records_df[fields_to_keep].copy()


def filter_publish_fields(records_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fields needed for publish updates.
        Args:
            record: Records to filter.
    
        Returns:
            Records containing only publish fields.
    """
    fields_to_keep = [
        field for field in records_df.columns
        if field in PUBLISH_FIELDS
    ]

    return records_df[fields_to_keep].copy()


def remove_pages(
    ingest_sheet: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Remove Page objects from an ingest sheet.

    Page records may be identified by either the taxonomy term name ``Page``
    or the term ID ``17``.

    Args:
        ingest_sheet: Islandora ingest sheet.
        logger: Optional logger.

    Returns:
        Ingest sheet with all Page records removed.

    Raises:
        KeyError: If the ingest sheet does not contain ``field_model``.
    """
    normalized_models = (
        ingest_sheet['field_model']
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    # As in process_model, a Page object may be represented by either its
    # taxonomy term name ("Page") or its term ID ("17"), so both are matched.
    page_mask = normalized_models.isin({'page', '17'})
    page_count = int(page_mask.sum())

    if logger:
        logger.info(
            "Removed %d Page record(s) from the ingest sheet.",
            page_count,
        )

    return (
        ingest_sheet.loc[~page_mask]
        .reset_index(drop=True)
    )


# --- Mapping Helpers ---

def is_repeatable_field(field: str | None) -> bool:
    """Return whether an Islandora field is repeatable."""
    if not field:
        return False

    match = FIELDS.loc[
        FIELDS['Field'] == field,
        'Repeatable',
    ]

    if match.empty:
        return False

    return match.iat[0] == 'TRUE'


def resolve_agent_data(csv_field: str) -> MappingResolution | None:
    """Resolve linked-agent taxonomy and prefix from a CSV field name.

    This only looks values up; any issue or transformation it finds is
    returned in the resolution so it can be logged later, per record.

    Args:
        csv_field: CSV field name.

    Returns:
        Resolution with the linked-agent mapping on success, a resolution
        describing the failure when the field looks like an agent field but
        cannot be resolved, or None if the field is not agent-like.
    """
    if not csv_field or '_' not in csv_field:
        return None

    # Agent-related CSV columns follow a "{role}_{agent_type}" naming
    # convention in the metadata sheet, e.g. "photographer_person" or
    # "sponsor_corporate". Splitting on the last underscore separates the
    # relator role (which may itself contain underscores) from the agent
    # type suffix.
    role_term, agent_type = csv_field.rsplit('_', 1)
    role_term = role_term.replace('_', ' ')

    relator = RELATOR_TERMS.get(role_term)

    if not relator:
        return MappingResolution(
            agent_role_term=role_term,
            agent_issue="could not find relator term",
        )

    relator_code = relator['code']
    relator_note = relator['note'] or None

    agent_type = LINKED_AGENT_TYPES.get(agent_type)

    if not agent_type:
        return MappingResolution(
            agent_role_term=role_term,
            agent_issue="could not determine linked agent type",
            relator_code=relator_code,
            relator_note=relator_note,
        )

    return MappingResolution(
        mapping=FieldMapping(
            field='field_linked_agent',
            taxonomy=agent_type.replace('_', ' ').title(),
            prefix=f'relators:{relator_code}:{agent_type}:',
            repeatable=is_repeatable_field('field_linked_agent'),
        ),
        agent_role_term=role_term,
        relator_code=relator_code,
        relator_note=relator_note,
    )


@lru_cache(maxsize=None)
def resolve_column_mapping(csv_field: str) -> MappingResolution:
    """Resolve a source CSV field to its Islandora field mapping.

    Searches the template, manifest, and MARC field mapping tables for the
    supplied CSV field. If no direct mapping is found, linked-agent fields
    are handled separately, then fields that are themselves I2 fields.

    The result depends only on the field name and the mapping tables, which
    are fixed for the run, so it is cached. This function never logs.

    Args:
        csv_field: Source CSV field.

    Returns:
        MappingResolution containing the FieldMapping and any details to
        log. If no mapping is found, the FieldMapping has all optional
        attributes set to None and ``repeatable`` False.
    """
    if csv_field in SCAN_BATCH_DIR_FIELDS:
        return MappingResolution()

    mapping_columns = [
        'machine_name',
        'taxonomy',
        'prefix',
    ]

    for mapping_df in (
        TEMPLATE_FIELD_MAPPING,
        MANIFEST_FIELD_MAPPING,
        MARC_FIELD_MAPPING,
    ):
        match = mapping_df.loc[
            mapping_df['field'] == csv_field,
            mapping_columns,
        ]

        if match.empty:
            continue

        row = match.iloc[0]

        mapped_field = (
            None
            if pd.isna(row['machine_name'])
            else row['machine_name']
        )
        taxonomy = (
            None
            if pd.isna(row['taxonomy'])
            else row['taxonomy']
        )
        prefix = (
            None
            if pd.isna(row['prefix'])
            else row['prefix'].strip()
        )

        # Check if prefix contains capital letters and, thus, is a display label
        # requiring a trailing space
        if prefix:
            has_capitals = any(char.isupper() for char in prefix)
            prefix = f"{prefix} " if has_capitals else prefix

        return MappingResolution(
            mapping=FieldMapping(
                field=mapped_field,
                taxonomy=taxonomy,
                prefix=prefix,
                repeatable=is_repeatable_field(mapped_field),
            )
        )

    agent_resolution = None

    if check_if_agent_field(csv_field):
        agent_resolution = resolve_agent_data(csv_field)

        if agent_resolution and agent_resolution.mapping.is_mapped:
            return agent_resolution

    # Keep any failed agent lookup details so they are still reported
    base = agent_resolution or MappingResolution()

    # TODO: Support taxonomy and prefix matching by updating I2 field schema
    match = FIELDS.loc[FIELDS['Field'] == csv_field]

    if not match.empty:
        return replace(
            base,
            mapping=FieldMapping(
                field=csv_field,
                taxonomy=None,
                prefix=None,
                repeatable=is_repeatable_field(csv_field),
            ),
            fallback='i2_field',
        )

    return replace(base, fallback='unmapped')


def log_mapping_resolution(
    result: ProcessingResult,
    pid: str,
    csv_field: str,
    data: str,
    resolution: MappingResolution,
) -> None:
    """Log the issues and transformations found while resolving a field.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        csv_field: Source CSV field.
        data: Source cell value.
        resolution: Cached resolution for the field.
    """
    if resolution.relator_note:
        result.log_transformation(
            pid,
            csv_field,
            resolution.agent_role_term,
            resolution.relator_code,
            resolution.relator_note,
        )
        logging.getLogger(LOGGER_NAME).warning(
            "Record %s: Relator code %s - %s",
            pid,
            resolution.relator_code,
            resolution.relator_note,
        )

    if resolution.agent_issue:
        result.log_issue(
            pid,
            csv_field,
            resolution.agent_role_term,
            resolution.agent_issue,
        )

    if resolution.fallback == 'i2_field':
        result.log_issue(
            pid,
            csv_field,
            data,
            "input field is an I2 field; ensure taxonomy term and prefix are correct",
            (
                "Input field is an I2 field; ensure taxonomy term and prefix "
                "are correct."
            ),
        )
    elif resolution.fallback == 'unmapped':
        result.log_issue(
            pid,
            csv_field,
            data,
            "could not find matching I2 field",
            (
                f"Could not find matching I2 field for CSV field "
                f"'{csv_field}' with data '{data}'. Dropping column."
            ),
        )


def get_mapped_field(
    result: ProcessingResult,
    pid: str,
    csv_field: str,
    data: str,
) -> FieldMapping:
    """Return the Islandora field mapping for a source CSV field.

    The lookup itself is cached per field name. Issues and transformations
    found during the lookup are logged for this record only when the cell
    has data, so blank cells in unmapped or special columns don't add noise
    to the reports.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        csv_field: Source CSV field.
        data: Source cell value.

    Returns:
        FieldMapping object containing the mapped Islandora field, taxonomy,
        prefix, and repeatability. If no mapping is found, all optional
        attributes will be None and ``repeatable`` will be False.
    """
    resolution = resolve_column_mapping(csv_field)

    if not pd.isna(data) and str(data).strip():
        log_mapping_resolution(result, pid, csv_field, data, resolution)

    return resolution.mapping


# --- Record Building Helpers ---

def add_value(
    result: ProcessingResult,
    record: dict,
    csv_field: str | None,
    field: str | None,
    value: str,
    prefix: str | None = None,
) -> str | None:
    """Add a processed value to a record field.

    Args:
        result: Runtime processing result.
        record: Record dictionary being updated.
        csv_field: Source CSV field.
        field: Target machine field.
        value: Raw value to add.
        prefix: Optional value prefix.

    Returns:
        Processed value added to the record, or None.
    """
    if not field:
        log_msg = (
            f"Missing I2 field for value '{value}' from CSV field "
            f"'{csv_field}'."
        )
        result.log_issue(
            record['id'][0],
            None,
            value,
            f"missing I2 field for value from CSV field {csv_field}",
            log_msg,
        )
        return None

    if field in FORMATTED_FIELDS:
        value = remove_whitespaces(text=value, allow_newlines=True) 
    else:
        value = remove_whitespaces(value)

    if not value:
        return value
    
    unprefixed_value = value

    if prefix:
        value = f'{prefix}{value}'

    values = record.get(field, [])

    if field == 'field_linked_agent':
        # Linked agents use a different dedup rule than other controlled
        # fields: the same person/entity name can legitimately appear more
        # than once, as long as each appearance has a different role prefix
        # (e.g. "photographer:Jane Doe" and "editor:Jane Doe" are both kept,
        # since they represent different relationships to the object). Only
        # an exact repeat of the same prefix + name is treated as a
        # duplicate.
        if value not in values:
            values.append(value)
    else:
        # Prevent duplicates with and without prefix. When the same term
        # appears both prefixed and unprefixed, keep the prefixed form
        # regardless of which one arrived first.
        #
        # This matters because the same term can reach this function twice
        # for one record: once bare (e.g. from process_model deriving
        # "Still Image" directly) and once prefixed (e.g. from a mapped CSV
        # column that attaches a vocabulary prefix like "resource_type:").
        #
        # Stored entries are PrefixedValue objects, which carry their bare
        # term and prefix separately. Entries are compared on the bare term
        # directly rather than by parsing the combined string, so terms that
        # themselves contain colons are handled correctly.
        value = PrefixedValue(unprefixed_value, prefix)

        stored = next(
            (
                v for v in values 
                if getattr(v, 'term', v) == unprefixed_value
            ),
            None,
        )

        if stored is None:
            values.append(value)
        elif value.prefix and not getattr(stored, 'prefix', ''):
            # A bare form is stored; the prefixed form just arrived and
            # takes precedence, so replace the bare entry.
            values[values.index(stored)] = value
        # else: the incoming value is bare while a prefixed form is already
        # stored, it exactly repeats the stored value, or both are
        # differently prefixed variants (keep whichever was seen first).
        # In every case, drop the incoming value.
    record[field] = values

    return value


def add_title(
    result: ProcessingResult,
    record: dict,
    value: str,
) -> dict:
    """Add a title value to the record.

    Args:
        result: Runtime processing result.
        record: Record to update.
        value: Title value.

    Returns:
        Updated record.
    """
    add_value(result, record, None, 'title', value)
    add_value(result, record, None, 'field_full_title', value)

    return record


def process_title(
    result: ProcessingResult,
    record: dict,
    title_parts: dict,
) -> str | None:
    """Build and add a formatted title from title parts.

    Args:
        result: Runtime processing result.
        record: Record being updated.
        title_parts: Title components, keyed by source CSV field name.
            Recognized keys are ``title``, ``volume``, and ``issue`` (the
            metadata template's header). ``number`` is also accepted as an
            alternate name for the issue.

    Returns:
        Formatted title, if created.
    """
    title = title_parts.get('title')

    if title:
        volume = title_parts.get('volume')
        issue = title_parts.get('issue') or title_parts.get('number')

        if volume:
            title += f", vol. {volume}"

        if issue:
            title += f", no. {issue}"

        add_title(result, record, title)

    return title


# --- Validation Helpers ---

@lru_cache(maxsize=500)
def _check_network_status(value: str) -> tuple[bool, str]:
    """Perform cached HTTP validation for a collection node ID.

    Args:
        value: Node ID to check.

    Returns:
        Tuple containing validation status and error description.
    """
    url = f'https://digital.library.pitt.edu/node/{value}'

    try:
        response = requests.head(url, allow_redirects=True, timeout=10)

        if response.status_code == 404: # Page Not Found - node doesn't exist
            return False, f"URL returned 404: {url}"

        if response.status_code in {
            200, # OK - published
            301, # Moved Permanently - redirect to published collection page
            403, # Forbidden - unpublised collection page
        }:
            return True, ''

        return False, f"Unexpected status code {response.status_code}: {url}"

    except requests.exceptions.RequestException as error:
        msg = "HTTP request failed"
        logging.getLogger(LOGGER_NAME).exception(msg)

        return False, f"{msg}: {str(error)}"


def validate_collection_id(
    result: ProcessingResult,
    pid: str,
    value: str,
) -> bool:
    """Validate a collection node ID.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        value: Collection node ID.

    Returns:
        True if the collection node ID resolves; otherwise False.
    """
    is_valid, error_msg = _check_network_status(value)

    # Call the cached network function
    if not is_valid:
        result.log_issue(
            pid,
            'field_member_of',
            value,
            f"invalid collection node ID: {error_msg}",
        )

    return is_valid


def validate_domain(
    result: ProcessingResult,
    pid: str,
    value: str,
) -> bool:
    """Validate a domain access value.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        value: Domain access value.

    Returns:
        True if valid; otherwise False.
    """
    if value in DOMAINS:
        return True

    result.log_issue(
        pid,
        'field_domain_access',
        value,
        "invalid domain",
    )

    return False


def validate_coordinates(
    result: ProcessingResult,
    pid: str,
    value: str,
) -> bool:
    """Validate geographical coordinates.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        value: Coordinate string.

    Returns:
        True if coordinates are valid; otherwise False.
    """
    def fail(message: str) -> bool:
        result.log_issue(
            pid,
            'field_coordinates',
            value,
            f"invalid coordinates: {message}",
        )
        return False

    if not isinstance(value, str):
        return fail("coordinate value must be a string")

    cleaned = remove_whitespaces(value)

    if not cleaned:
        return fail("coordinate value is empty")

    # Expect exactly two parts: latitude, longitude
    parts = re.split(r'\s*[;,]\s*', cleaned)

    if len(parts) != 2:
        return fail("expected two coordinates separated by ',' or ';'")

    lat_str, lon_str = parts

    def parse_decimal(token: str) -> float | None:
        """Parse a decimal degree token."""
        if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', token):
            return None

        try:
            return float(token)
        except ValueError:
            return None

    # DMS (Degrees, Minutes, Seconds) pattern
    dms_pattern = re.compile(
        r'''
        ^\s*
        (?P<deg>[+-]?\d+(?:\.\d+)?)
        \s*[°ºd]?\s*
        (?:
            (?P<min>\d+(?:\.\d+)?)\s*['m]?\s*
        )?
        (?:
            (?P<sec>\d+(?:\.\d+)?)\s*["s]?\s*
        )?
        (?P<hem>[NnSsEeWw])?
        \s*$
        ''',
        re.VERBOSE,
    )

    def parse_dms(token: str) -> float | None:
        """Parse DMS coordinate to decimal degrees."""
        match = dms_pattern.match(token)

        if not match:
            return None

        try:
            degrees = float(match.group('deg'))
            minutes = (
                float(match.group('min'))
                if match.group('min')
                else 0.0
            )
            seconds = (
                float(match.group('sec'))
                if match.group('sec')
                else 0.0
            )
            hemisphere = match.group('hem')
        except Exception:
            return None

        sign = -1.0 if degrees < 0 else 1.0
        degrees = abs(degrees)
        decimal_degrees = degrees + minutes / 60 + seconds / 3600
        decimal_degrees *= sign

        if hemisphere:
            hemisphere = hemisphere.upper()

            if hemisphere in ('S', 'W'):
                decimal_degrees = -abs(decimal_degrees)
            else:
                decimal_degrees = abs(decimal_degrees)

        return decimal_degrees

    def parse_one(token: str) -> float | None:
        """Parse a decimal or DMS coordinate token."""
        decimal_value = parse_decimal(token)

        if decimal_value is not None:
            return decimal_value

        return parse_dms(token)

    lat = parse_one(lat_str)
    lon = parse_one(lon_str)

    if lat is None or lon is None:
        return fail("invalid coordinates; must be decimal or sexagesimal")

    if not (-90 <= lat <= 90):
        return fail("latitude is out of range")

    if not (-180 <= lon <= 180):
        return fail("longitude is out of range")

    return True


def validate_edtf_date(
    result: ProcessingResult,
    pid: str,
    field: str,
    value: str,
) -> bool:
    """Validate an EDTF date.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        field: Source field name.
        value: Date value.

    Returns:
        True if valid; otherwise False.
    """
    try:
        edtf_date = parse_edtf(value)

        if not edtf_date:
            result.log_issue(
                pid,
                field,
                value,
                "invalid EDTF date",
            )
            return False

        return True

    except Exception:
        valid = bool(DRUPAL_EXTENDED_EDTF_PATTERN.search(value))

        if not valid:
            result.log_issue(
                pid,
                field,
                value,
                "could not parse EDTF date",
                f"Could not parse EDTF date '{value}'.",
            )
            return False

        return True


def build_taxonomy_lookup(taxonomies: pd.DataFrame) -> TaxonomyLookup:
    """Index taxonomy terms by vocabulary.

    Args:
        taxonomies: Taxonomy table with ``Vocabulary``, ``Name``, and
            ``Term ID`` columns.

    Returns:
        TaxonomyLookup built from the table.
    """
    names = {}
    term_ids = {}
    id_to_name = {}

    for vocabulary, name, term_id in zip(
        taxonomies['Vocabulary'],
        taxonomies['Name'],
        taxonomies['Term ID'],
    ):
        names.setdefault(vocabulary, set()).add(name)
        term_ids.setdefault(vocabulary, set()).add(term_id)
        id_to_name.setdefault(vocabulary, {}).setdefault(term_id, name)

    return TaxonomyLookup(
        names={vocab: frozenset(values) for vocab, values in names.items()},
        term_ids={
            vocab: frozenset(values) for vocab, values in term_ids.items()
        },
        id_to_name=id_to_name,
    )


@lru_cache(maxsize=1)
def get_taxonomy_lookup() -> TaxonomyLookup:
    """Return the taxonomy index, building it on first use.

    The index is built from the loaded taxonomy table. Call
    ``get_taxonomy_lookup.cache_clear()`` if the table is reloaded.
    """
    return build_taxonomy_lookup(TAXONOMIES)


@lru_cache(maxsize=None)
def warn_unknown_taxonomy(taxonomy: str) -> None:
    """Log once that a taxonomy name matches no loaded vocabulary."""
    logging.getLogger(LOGGER_NAME).warning(
        "Taxonomy '%s' was not found among the loaded vocabularies; every "
        "term checked against it will be reported as missing.",
        taxonomy,
    )


def validate_term(
    result: ProcessingResult,
    pid: str,
    field: str,
    value: str,
    taxonomy: str,
) -> bool:
    """Validate that a term exists in a taxonomy.

    Args:
        result: Runtime processing result.
        pid: Current record PID.
        field: Source field name.
        value: Term value.
        taxonomy: Taxonomy name.

    Returns:
        True if term is valid; otherwise False.
    """
    lookup = get_taxonomy_lookup()

    if taxonomy not in lookup.names:
        warn_unknown_taxonomy(taxonomy)

    found = (
        value in lookup.names.get(taxonomy, ())
        or value in lookup.term_ids.get(taxonomy, ())
    )

    if not found:
        result.log_issue(
            pid,
            field,
            value,
            f"could not find term in {taxonomy} taxonomy",
            f"Could not find term '{value}' in {taxonomy} taxonomy.",
        )
        return False

    return True


def validate_record(
    result: ProcessingResult,
    record: dict,
    ingest_sheet: pd.DataFrame,
    ingest_task: str,
    id_column: str,
) -> dict:
    """Validate fields and values in a metadata record.

    Args:
        result: Runtime processing result.
        record: Record to validate.
        ingest_sheet: Full ingest sheet.
        ingest_task: Ingest task.
        id_column: Ingest sheet column that holds the record identifier.

    Returns:
        Validated record.
    """
    pids = record.get('id', [])
    pid = pids[0] if pids else ''
    logger = logging.getLogger(LOGGER_NAME)

    for field, values in record.items():
        match = FIELDS.loc[FIELDS['Field'] == field]

        if match.empty:
            logger.warning(
                "Record %s: Field '%s' not found in FIELDS lookup.",
                pid,
                field,
            )
            continue

        field_manager = match.iloc[0]

        if (
            field_manager.Field_Type == 'Text (plain)'
            and field not in FILE_REFERENCE_COLUMNS
        ):
            for value in values:
                if len(value) > 255:
                    result.log_issue(
                        pid,
                        field,
                        value,
                        "value exceeds character limit",
                    )

        if field_manager.Field_Type == 'Number (integer)':
            for value in values:
                try:
                    int(value)
                except (ValueError, TypeError):
                    result.log_issue(
                        pid,
                        field,
                        value,
                        (
                            "expected an integer, but got "
                            f"{type(value).__name__}: {value}"
                        ),
                    )

        if field_manager.Repeatable == 'FALSE' and len(values) > 1:
            result.log_issue(
                pid,
                field,
                values,
                "multiple values in nonrepeatable field",
            )

    if ingest_task != 'create':
        return record

    for field in MANDATORY_FIELDS:
        parent_id = record.get('parent_id')

        # Child records (records with a parent_id, e.g. individual pages
        # within a Paged Content object) are held to different mandatory
        # field rules than top-level objects, since they inherit context
        # from their parent rather than needing it stated independently:
        if parent_id:
            parent_id = parent_id[0]

            # A child with no title of its own is given a fallback title
            # equal to its own identifier, rather than being flagged as
            # missing a required field.
            if field == 'title' and not record[field]:
                add_value(result, record, None, 'title', pid)

            # Domain access isn't set independently per child; it's
            # inherited from whichever domain(s) the parent object belongs
            # to, so children stay visible on the same site(s) as their
            # parent.
            elif field == 'field_domain_access' and not record[field]:
                parent_domains = get_parent_values(
                    ingest_sheet,
                    pid,
                    parent_id,
                    'field_domain_access',
                    id_column,
                )

                for domain in parent_domains:
                    add_value(
                        result,
                        record,
                        None,
                        'field_domain_access',
                        domain,
                    )
                    
            # The depositor isn't set independently per child either; it's
            # inherited from the parent object, so a child is attributed to
            # the same depositor as its parent.
            elif field == 'field_depositor' and not record[field]:
                parent_depositors = get_parent_values(
                    ingest_sheet,
                    pid,
                    parent_id,
                    'field_depositor',
                    id_column,
                )

                # Apply the same vocabulary prefix the mapping tables give a
                # depositor entered directly on a record (e.g.
                # "contributing_institution:"), so inherited values match.
                depositor_mapping = get_mapped_field(
                    result,
                    pid,
                    'field_depositor',
                    '',
                )

                for depositor in parent_depositors:
                    add_value(
                        result,
                        record,
                        None,
                        'field_depositor',
                        depositor,
                        depositor_mapping.prefix,
                    )

            # Skip since children do not get collection associations
            elif field == 'field_member_of':
                continue

        missing_value = len(record[field]) < 1

        if missing_value:
            if (
                field == 'id'
                and record.get('node_id')
            ):
                continue

            result.log_issue(
                pid,
                field,
                None,
                "record missing required field",
                f"Missing required field {field}.",
            )

    return record


# --- Field-Specific Processing ---

def get_parent_values(
    ingest_sheet: pd.DataFrame,
    pid: str,
    parent_id: str,
    column: str,
    id_column: str,
) -> list[str]:
    """Inherit the values of a field from a parent record.

    Args:
        ingest_sheet: Master ingest DataFrame.
        pid: PID of the current child record.
        parent_id: PID of the parent record.
        column: Ingest sheet column to read from the parent row (e.g.
            ``field_domain_access`` or ``field_depositor``).
        id_column: Ingest sheet column that holds the record identifier.

    Returns:
        Parent values for the column, if found.
    """
    parent_values = []

    try:
        # Locate parent row and extract the requested column
        match = ingest_sheet.loc[
            ingest_sheet[id_column] == parent_id,
            column,
        ]

        # Tokenize the values
        if not match.empty and pd.notna(match.values[0]):
            extra_delimiters = (
                ',' if column in COMMA_DELIMITED_FIELDS else ''
            )
            parent_values = split_and_clean(
                str(match.values[0]),
                extra_delimiters=extra_delimiters,
            )

    except Exception:
        logging.getLogger(LOGGER_NAME).exception(
            "Record %s: Failed to retrieve parent %s.",
            pid,
            column,
        )

    return parent_values


def process_model(
    result: ProcessingResult,
    record: dict,
    field: str,
    value: str,
) -> bool:
    """Validate and process an object model.

    Args:
        result: Runtime processing result.
        record: Record being processed.
        field: Source field name.
        value: Object model value.

    Returns:
        True if model is valid; otherwise False.
    """
    pid = record['id'][0]

    # Source data may supply either the model's display name (e.g. "Page")
    # or its taxonomy term ID (e.g. "17") - both are valid inputs throughout
    # this pipeline (see also the similar name-or-ID check in remove_pages).
    # Try the name first, since that's the common case, then fall back to
    # resolving a term ID to its name via the taxonomy lookup.
    model_mapping = MODEL_MAPPING.get(value)

    if not model_mapping:
        model_name = (
            get_taxonomy_lookup()
            .id_to_name
            .get('Islandora Models', {})
            .get(value)
        )
        if model_name is not None:
            model_mapping = MODEL_MAPPING.get(model_name)

    if model_mapping is None:
        result.log_issue(
            pid,
            'field_model',
            value,
            "could not find term in model taxonomy",
        )
        return False

    resource_type = model_mapping.get('resource_type')
    add_value(
        result,
        record,
        field,
        'field_resource_type',
        resource_type,
    )

    display_hint = model_mapping.get('display_hint')
    add_value(
        result,
        record,
        field,
        'field_display_hints',
        display_hint,
    )

    return True


# --- Main Workflow ---

def process_record(
    result: ProcessingResult,
    row: pd.Series,
    index: int,
    id_column: str,
) -> dict | None:
    """Transform a raw CSV row into a structured metadata record.

    Args:
        result: Runtime processing result.
        row: Source row.
        index: Row index.
        id_column: Column that holds the record identifier.

    Returns:
        Processed record, or None if processing fails.
    """
    # Setup record
    record = initialize_record()

    # Read the record identifier from the designated column
    pid = row[id_column] if id_column in row.index else None
    pid = remove_whitespaces(str(pid)) if pd.notna(pid) else None

    if not pid:
        msg = f"row {index} missing required identifier"
        result.log_issue(
            'UNKNOWN',
            id_column,
            None,
            msg,
            f"{msg.capitalize()}.",
        )
        return None

    try:
        add_value(result, record, None, 'id', str(pid))

        # Process values in each field
        title_parts = {}

        for csv_field, data in row.items():
            # Confirm that input field is mapped and data exists
            mapping = get_mapped_field(
                result,
                pid,
                csv_field,
                data,
            )

            if not mapping.is_mapped or pd.isna(data):
                continue

            if mapping.field == 'id':
                continue

            if mapping.repeatable:
                extra_delimiters = (
                    ',' if mapping.field in COMMA_DELIMITED_FIELDS else ''
                )
                values = split_and_clean(data, extra_delimiters=extra_delimiters)
            elif mapping.field in FORMATTED_FIELDS:
                cleaned = remove_whitespaces(data, allow_newlines=True)
                values = [cleaned] if cleaned else []
            else:
                cleaned = remove_whitespaces(data)
                values = [cleaned] if cleaned else []

            for value in values:
                if mapping.field == 'field_full_title':
                    title_parts[csv_field] = value
                    continue

                if mapping.field == 'field_model':
                    process_model(result, record, csv_field, value)
                elif mapping.field == 'field_member_of':
                    validate_collection_id(result, pid, value)
                elif mapping.field == 'field_domain_access':
                    validate_domain(result, pid, value)
                elif mapping.field == 'field_coordinates':
                    validate_coordinates(result, pid, value)
                elif mapping.field in DATE_FIELDS:
                    validate_edtf_date(result, pid, csv_field, value)
                elif mapping.field in CONTROLLED_FIELDS:
                    validate_term(
                        result,
                        pid,
                        csv_field,
                        value,
                        mapping.taxonomy,
                    )

                if value:
                    add_value(
                        result,
                        record,
                        csv_field,
                        mapping.field,
                        value,
                        mapping.prefix,
                    )

        process_title(result, record, title_parts)

    except Exception:
        logging.getLogger(LOGGER_NAME).exception(
            "An error occurred while processing record %s.",
            pid,
        )
        return None

    return record


def log_family_issues(
    result: ProcessingResult,
    pid: str,
    position: int,
    family_order: FamilyOrder,
    failed_keys: set[str],
) -> None:
    """Report parent/child problems found while ordering records.

    The record is still written in every case; these are issues to review.

    Args:
        result: Runtime processing result.
        pid: PID of the record being processed.
        position: Position of the record in the reordered sheet.
        family_order: Result of ordering the records by family.
        failed_keys: Identifiers of records that failed to process.
    """
    parent = family_order.parents[position]

    if position in family_order.circular:
        result.log_issue(
            pid,
            'parent_id',
            parent,
            "circular parent_id reference",
            (
                f"parent_id '{parent}' leads back to this record. The record "
                "was written, but it cannot be ordered after its parent."
            ),
        )
    elif position in family_order.unresolved_parents:
        result.log_issue(
            pid,
            'parent_id',
            parent,
            "parent not found in the ingest sheet",
            (
                f"Parent '{parent}' was not found in the ingest sheet. The "
                "record was written anyway."
            ),
        )
    elif parent and parent in failed_keys:
        result.log_issue(
            pid,
            'parent_id',
            parent,
            "parent record failed to process",
            (
                f"Parent '{parent}' failed to process and is not in the "
                "ingest file. The record was written anyway."
            ),
        )


def process_files(
    export_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    config: AppConfig,
    result: ProcessingResult,
) -> bool:
    """Process records, write ingest batches, and generate reports.

    Args:
        export_df: Export sheet DataFrame.
        metadata_df: Metadata DataFrame.
        config: Application configuration object.
        result: Runtime processing result.

    Returns:
        True if processing completes; otherwise False.
    """
    logger = logging.getLogger(LOGGER_NAME)
    completed = False

    try:
        # Confirm the identifier column exists in the metadata sheet
        if not metadata_df.empty:
            validate_id_column(metadata_df, config.id_column)
            logger.info(
                "Using '%s' as the identifier column.",
                config.id_column,
            )

        # Merge export and metadata sheets
        if not export_df.empty and not metadata_df.empty:
            ingest_sheet, unmatched_records = merge_sheets(
                export_df,
                metadata_df,
                config.id_column,
            )
        else:
            ingest_sheet = metadata_df
            unmatched_records = pd.DataFrame()
            
        # Write unmatched rows immediately, before any validation can stop
        # the run, so the report is always available for troubleshooting
        unmatched_log_csv = None

        if not unmatched_records.empty:
            unmatched_log_csv = (
                config.log_dir / f'{config.file_prefix}_unmatched.csv'
            )
            logger.warning(
                "Unmatched rows found, writing to %s.",
                unmatched_log_csv,
            )
            df_to_csv(unmatched_records, unmatched_log_csv)

        # Confirm that ingest sheet contains node_ids for update task
        if config.ingest_task == 'update':
            if 'node_id' not in ingest_sheet.columns:
                message = (
                    "Ingest sheet missing required 'node_id' column for update "
                    "task."
                )
                logger.error(message)
                raise ValueError(message)

            blank_node_ids = (
                ingest_sheet['node_id']
                .fillna('')
                .astype(str)
                .str.strip()
                .eq('')
            )

            if blank_node_ids.any():
                message = (
                    f"Update ingest contains {int(blank_node_ids.sum())} "
                    "record(s) without a node_id."
                )

                if unmatched_log_csv:
                    message += f" See unmatched report: {unmatched_log_csv}"

                logger.error(message)
                raise ValueError(message)

        # Add publication field
        publish_value = '1' if config.publish else '0'
        ingest_sheet['published'] = publish_value

        if config.metadata_level == 'publish':
            ingest_sheet = filter_publish_fields(ingest_sheet)

        # Remove Page objects
        elif config.remove_pages:
            ingest_sheet = remove_pages(ingest_sheet, logger)

        # Process Batch
        buffer = []
        result.current_batch = 1

        if config.metadata_level == 'publish':
            records = ingest_sheet.to_dict('records')

            for record in tqdm(
                records,
                total=len(records),
                desc="Processing Records",
                unit='record',
            ):
                buffer.append(record)

                if should_flush_batch(buffer, config.batch_size):
                    flush_batch(buffer, result.current_batch, config)
                    result.advance_batch()
                    buffer.clear()

        else:
            # Order rows so parents precede children and families stay whole
            family_order = order_records_by_family(
                ingest_sheet,
                config.id_column,
                use_hierarchy=True,
            )
            ingest_sheet = family_order.sheet
            failed_keys = set()

            for position, (idx, row) in enumerate(
                tqdm(
                    ingest_sheet.iterrows(),
                    total=len(ingest_sheet),
                    desc="Processing Records",
                    unit='record',
                )
            ):
                record = process_record(result, row, idx, config.id_column)

                if record:
                    log_family_issues(
                        result,
                        record['id'][0],
                        position,
                        family_order,
                        failed_keys,
                    )
                    record = validate_record(
                        result,
                        record,
                        ingest_sheet,
                        config.ingest_task,
                        config.id_column,
                    )
                    record = format_record(record)
                    buffer.append(record)
                else:
                    result.increment_failure_count()

                    if family_order.keys[position]:
                        failed_keys.add(family_order.keys[position])

                # Complete batch once the size is reached, but only at the
                # end of a family so parents and children are never split
                if (
                    buffer
                    and family_order.family_end[position]
                    and should_flush_batch(buffer, config.batch_size)
                ):
                    if len(buffer) > config.batch_size:
                        logger.info(
                            "Batch %d has %d records, more than the batch "
                            "size of %d, to keep parent and child records "
                            "together.",
                            result.current_batch,
                            len(buffer),
                            config.batch_size,
                        )

                    flush_batch(
                        buffer,
                        result.current_batch,
                        config,
                    )

                    result.advance_batch()
                    buffer.clear()

        # Flush remaining buffer
        if buffer:
            flush_batch(
                buffer,
                result.current_batch,
                config,
            )

        completed = True

        return True

    except Exception as error:
        logger.exception("An error occurred during processing.")
        print(
            f"\n{ERROR_SYMBOL} Data processor stopped unexpectedly: "
            f"{error}.",
            f"See logs: {config.log_path}",
        )

        return False

    finally:
        # Write the issue and transformation reports even if processing
        # stopped early, so the findings collected so far aren't lost. When
        # the run failed before anything was collected, skip the reports to
        # avoid a misleading "no exceptions" message.
        if completed or result.issues or result.transformations:
            try:
                write_reports(
                    config.log_dir,
                    config.timestamp,
                    'metadata',
                    result.transformations,
                    result.issues
                )
            except Exception:
                logger.exception("Failed to write processing reports.")


def main() -> None:
    """Run the metadata ingest generation workflow."""
    logger = None
    config = None

    try:
        config = parse_arguments()

        # Set up output files
        config.batch_path = Path(config.batch_path)
        config.batch_dir = config.batch_path.name
        config.timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        config.file_prefix = f'{config.batch_dir}_{config.timestamp}'

        config.output_dir = create_directory(config.batch_path / 'import')
        config.log_dir = create_directory(config.batch_path / 'logs')
        config.log_path = (
            config.log_dir /
            f'{config.file_prefix}_ingest_{config.metadata_level}_processing.log'
        )

        setup_batch_directory(config.batch_path)

        # Set up logging
        logger = setup_logger(LOGGER_NAME, config.log_path)

        # Log configuration input
        config.log_configuration(logger)

        # Load taxonomies
        load_taxonomies(
            refresh=config.refresh_taxonomies,
            logger=logger,
        )

        # Index the loaded taxonomies for fast term validation
        get_taxonomy_lookup.cache_clear()
        get_taxonomy_lookup()

        result = ProcessingResult()

        # Convert export and metadata sheets to DataFrames
        export_df, metadata_df = load_input_sheets(config)

        # Process records and write ingest batches
        success = process_files(
            export_df,
            metadata_df,
            config,
            result,
        )

        critical_failures_count = result.unexpected_failure_count

    except Exception:
        msg = "A critical system error occurred during execution."

        if logger:
            logger.exception(msg)

        log_path = getattr(config, 'log_path', None)

        print(f"\n{ERROR_SYMBOL} {msg}")

        if log_path:
            print(f"See logs: {log_path}")
        else:
            traceback.print_exc()

        sys.exit(1)

    # A run that stopped early is a failure regardless of how many records
    # had already failed individually
    if not success:
        msg = "Processing did not complete successfully."

        if critical_failures_count > 0:
            unit = "record" if critical_failures_count == 1 else "records"
            msg += (
                f" {critical_failures_count} {unit} failed before the run "
                "stopped."
            )

        logger.error("%s", msg)
        sys.exit(1)

    if critical_failures_count > 0:
        unit = "record" if critical_failures_count == 1 else "records"
        msg = f"{critical_failures_count} {unit} failed to process."
        logger.error("%s", msg)
        print(
            f"\n{WARNING_SYMBOL} {msg}",
            f" See logs: {config.log_path}" if config.log_path else ""
        )
    else:
        msg = "All records were processed successfully."
        logger.info("%s", msg)
        print(f"\n{SUCCESS_SYMBOL} {msg}")


if __name__ == '__main__':
    main()

"""CSV Reader and Field Normalization Module.

Extracted and cleanly handle flexible CSV column headers,
dirty data parsing, and student-level record grouping.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO


# Column header aliases mapping standardized internal keys to expected CSV column names
ALIASES = {
    "student_id": ("student_id", "student id", "id", "learner_id"),
    "student_name": ("student_name", "student name", "name", "student"),
    "subject": ("subject", "course", "class", "course_name"),
    "score": ("score", "grade", "mark", "percentage", "performance"),
    "attendance": ("attendance", "attendance_rate", "attendance %", "present"),
    "remarks": ("remarks", "observation", "teacher_observation", "notes", "comment"),
    "date": ("date", "term", "semester", "period", "school_year"),
}


def normalize(value: Any) -> str:
    """Normalize strings/header names to lowercase alphanumeric underscore format.
    
    Example: 'Student ID %' -> 'student_id'
    """
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def get_field(row: dict[str, Any], name: str) -> Any:
    """Extract a field value from a CSV row by matching column aliases.
    
    Args:
        row: Dictionary representing a single CSV row.
        name: Standardized key name present in ALIASES (e.g. 'student_id').
        
    Returns:
        The matched field value string, or None if no matching header/value exists.
    """
    normalised_row = {normalize(key): value for key, value in row.items()}
    for alias in ALIASES[name]:
        key = normalize(alias)
        if key in normalised_row and str(normalised_row[key] or "").strip():
            return normalised_row[key]
    return None


def parse_number(value: Any) -> float | None:
    """Extract floating point numbers from messy string inputs (e.g. '92%', '88.5 marks')."""
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    try:
        return float(match.group()) if match else None
    except ValueError:
        return None


def parse_mean(values: Iterable[float]) -> float | None:
    """Calculate the arithmetic mean rounded to 2 decimal places."""
    vals = list(values)
    return round(sum(vals) / len(vals), 2) if vals else None


def read_student_csv(source: str | Path | TextIO) -> dict[str, list[dict[str, Any]]]:
    """Read a student records CSV file and group rows by student identifier.
    
    Args:
        source: File path, Path object, or file-like text stream.
        
    Returns:
        A dictionary mapping student identifiers to their list of raw CSV row dictionaries.
    """
    close_after = False
    if hasattr(source, "read"):
        stream = source
    else:
        stream = open(source, "r", encoding="utf-8-sig", newline="")
        close_after = True

    try:
        rows = list(csv.DictReader(stream))
    finally:
        if close_after:
            stream.close()

    if not rows:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        identifier = (
            get_field(row, "student_id")
            or get_field(row, "student_name")
            or f"student_{index}"
        )
        grouped[str(identifier).strip()].append(row)

    return dict(grouped)

read_student_csv_tool = {
  "name": "read_student_csv",
  "description": "Read a student records CSV file from a given file path, handle flexible column headers using internal aliases, parse dirty numeric values, and group rows by student identifier.",
  "parameters": {
    "type": "object",
    "properties": {
      "source": {
        "type": "string",
        "description": "The file path or path string to the student records CSV file."
      }
    },
    "required": [
      "source"
    ]
  }
}
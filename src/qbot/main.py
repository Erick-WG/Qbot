"""Student-record analysis and explainable study/career-path suggestions.

Usage: python -m qbot.main students.csv [--json]

The output is decision support, not a diagnosis or a prediction of a student's
future. Review recommendations with the student, family, and educators.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO


ALIASES = {
	"student_id": ("student_id", "student id", "id", "learner_id"),
	"student_name": ("student_name", "student name", "name", "student"),
	"subject": ("subject", "course", "class", "course_name"),
	"score": ("score", "grade", "mark", "percentage", "performance"),
	"attendance": ("attendance", "attendance_rate", "attendance %", "present"),
	"remarks": ("remarks", "observation", "teacher_observation", "notes", "comment"),
	"date": ("date", "term", "semester", "period", "school_year"),
}

# These are transparent text cues, not validated assessments of learning style.
OBSERVATION_CUES = {
	"collaborative": ("team", "group", "collaborat", "peer"),
	"hands_on": ("hands-on", "practical", "build", "experiment", "project"),
	"visual": ("visual", "diagram", "chart", "map", "draw"),
	"independent": ("independent", "self-directed", "works alone", "initiative"),
	"verbal_or_discussion": ("discussion", "debate", "explains", "presentation", "verbal"),
	"needs_support": ("support", "struggle", "difficulty", "needs help", "incomplete"),
	"engagement_or_curiosity": ("curious", "engaged", "enthusiastic", "asks questions", "motivated"),
}

CAREER_PATHS = (
	{"keywords": ("math", "mathematics", "physics", "statistics"), "skills": ["quantitative reasoning", "data literacy", "problem solving"], "careers": ["engineering", "data analysis", "actuarial science"]},
	{"keywords": ("biology", "chemistry", "science", "health"), "skills": ["scientific investigation", "careful observation", "research literacy"], "careers": ["health sciences", "biomedical research", "environmental science"]},
	{"keywords": ("english", "literature", "writing", "history", "language"), "skills": ["writing", "critical reading", "research and communication"], "careers": ["communications", "law-related study", "education", "public policy"]},
	{"keywords": ("art", "design", "music", "drama", "media"), "skills": ["creative practice", "portfolio development", "iterative design"], "careers": ["design", "media production", "performing arts"]},
	{"keywords": ("computing", "computer", "technology", "coding", "ict"), "skills": ["computational thinking", "programming", "digital problem solving"], "careers": ["software development", "cybersecurity", "IT"]},
	{"keywords": ("business", "economics", "accounting"), "skills": ["financial literacy", "analysis", "planning"], "careers": ["business", "finance", "entrepreneurship"]},
}


def _normalise(value: Any) -> str:
	return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _field(row: dict[str, Any], name: str) -> Any:
	normalised = {_normalise(key): value for key, value in row.items()}
	for alias in ALIASES[name]:
		key = _normalise(alias)
		if key in normalised and str(normalised[key] or "").strip():
			return normalised[key]
	return None


def _number(value: Any) -> float | None:
	if value is None:
		return None
	match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
	try:
		return float(match.group()) if match else None
	except ValueError:
		return None


def _mean(values: Iterable[float]) -> float | None:
	values = list(values)
	return round(sum(values) / len(values), 2) if values else None


@dataclass
class StudentProfile:
	student_id: str
	student_name: str | None
	subject_averages: dict[str, float]
	strongest_subjects: list[str]
	focus_subjects: list[str]
	performance_trends: dict[str, str]
	attendance_average: float | None
	attendance_note: str
	observation_signals: list[str]
	evidence: list[str] = field(default_factory=list)


class StudentAnalysisAgent:
	"""Convert CSV records into evidence-backed profiles and study suggestions."""

	def analyse(self, source: str | Path | TextIO) -> list[dict[str, Any]]:
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
			return []

		grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for index, row in enumerate(rows, start=1):
			identifier = _field(row, "student_id") or _field(row, "student_name") or f"student_{index}"
			grouped[str(identifier).strip()].append(row)

		results = []
		for identifier, records in grouped.items():
			profile = self._profile(identifier, records)
			results.append({"profile": asdict(profile), "recommendations": self._recommend(profile)})
		return results

	def _profile(self, identifier: str, records: list[dict[str, Any]]) -> StudentProfile:
		name = next((_field(row, "student_name") for row in records if _field(row, "student_name")), None)
		scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
		attendance_values, remarks = [], []
		for row in records:
			subject, score = _field(row, "subject"), _number(_field(row, "score"))
			if subject and score is not None:
				date = str(_field(row, "date") or "")
				scores[str(subject).strip()].append((score, date))
			attendance = _number(_field(row, "attendance"))
			if attendance is not None:
				attendance_values.append(attendance)
			remark = _field(row, "remarks")
			if remark:
				remarks.append(str(remark).lower())

		averages = {subject: round(sum(score for score, _ in values) / len(values), 2)
					for subject, values in scores.items()}
		ranked = sorted(averages, key=averages.get, reverse=True)
		strongest = [subject for subject in ranked if averages[subject] >= 70][:3]
		focus = [subject for subject in sorted(averages, key=averages.get) if averages[subject] < 60][:3]
		trends = {}
		for subject, values in scores.items():
			dated = sorted(values, key=lambda item: item[1]) if all(item[1] for item in values) else values
			if len(dated) >= 2:
				delta = dated[-1][0] - dated[0][0]
				trends[subject] = "improving" if delta >= 5 else "declining" if delta <= -5 else "stable"

		all_remarks = " ".join(remarks)
		signals = [label for label, cues in OBSERVATION_CUES.items()
				   if any(cue in all_remarks for cue in cues)]
		attendance = _mean(attendance_values)
		if attendance is None:
			attendance_note = "No attendance data provided"
		elif attendance <= 1:
			attendance_note = "Low attendance" if attendance < 0.85 else "Attendance appears consistent"
		else:
			attendance_note = "Low attendance" if attendance < 85 else "Attendance appears consistent"

		evidence = [f"{len(records)} record(s) analysed"]
		if not averages:
			evidence.append("No numeric subject scores found; confirm CSV score and subject column names")
		if not remarks:
			evidence.append("No teacher observation remarks provided")
		if attendance is None:
			evidence.append("No attendance values provided")
		return StudentProfile(str(identifier), str(name) if name else None, averages, strongest, focus,
							  trends, attendance, attendance_note, signals, evidence)

	def _recommend(self, profile: StudentProfile) -> dict[str, Any]:
		focus = profile.focus_subjects or (profile.strongest_subjects[:1] if profile.strongest_subjects else [])
		suggested_skills, careers = [], []
		interests = " ".join(profile.strongest_subjects).lower()
		for path in CAREER_PATHS:
			if any(word in interests for word in path["keywords"]):
				suggested_skills.extend(path["skills"])
				careers.extend(path["careers"])
		actions = [f"Set a short-term learning goal and review progress in {subject}." for subject in focus]
		if profile.attendance_note == "Low attendance":
			actions.append("Discuss attendance barriers and supportive next steps with the student and school.")
		if "collaborative" in profile.observation_signals:
			actions.append("Consider structured peer-learning opportunities, based on teacher observations.")
		if "hands_on" in profile.observation_signals:
			actions.append("Consider practical or project-based practice, based on teacher observations.")
		if not actions:
			actions.append("Gather student interests and additional assessment evidence before setting targets.")
		return {
			"focus_subjects": focus,
			"skills_to_build": list(dict.fromkeys(suggested_skills)),
			"possible_related_career_areas": list(dict.fromkeys(careers)),
			"suggested_actions": actions,
			"rationale": "Career areas are broad possibilities linked to current stronger subjects, not predicted outcomes.",
		}


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyse student CSV records")
	parser.add_argument("csv_file", type=Path)
	parser.add_argument("--json", action="store_true", help="Emit JSON (default output is also JSON)")
	args = parser.parse_args()
	try:
		results = StudentAnalysisAgent().analyse(args.csv_file)
	except (OSError, csv.Error) as error:
		parser.error(str(error))
	print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()

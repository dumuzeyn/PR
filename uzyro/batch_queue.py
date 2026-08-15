from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Callable

from .automation import ActionRunner, SkipBatchFile, load_action
from .core import Document


JOB_STATES = {"pending", "running", "completed", "completed_with_errors", "cancelled", "failed"}


@dataclass
class BatchItem:
    source: str
    target: str = ""
    state: str = "pending"
    message: str = ""
    executed_steps: int = 0


@dataclass
class BatchJob:
    action: dict[str, Any]
    sources: list[str]
    destination: str
    suffix: str = ".png"
    conflict: str = "rename"
    on_error: str = "continue"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "pending"
    items: list[BatchItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        if not self.items:
            self.items = [BatchItem(source) for source in self.sources]

    @property
    def completed_count(self) -> int:
        return sum(item.state in {"completed", "skipped", "failed"} for item in self.items)

    @property
    def error_count(self) -> int:
        return sum(item.state == "failed" for item in self.items)


class BatchQueue:
    def __init__(self, runner: ActionRunner | None = None) -> None:
        self.runner = runner or ActionRunner()
        self.jobs: list[BatchJob] = []
        self._cancel = threading.Event()
        self._lock = threading.RLock()

    def enqueue(
        self,
        action: str | Path | dict[str, Any],
        sources: list[str | Path],
        destination: str | Path,
        *,
        suffix: str = ".png",
        conflict: str = "rename",
        on_error: str = "continue",
    ) -> BatchJob:
        if conflict not in {"rename", "overwrite", "skip"}:
            raise ValueError("Режим совпадения имён должен быть rename, overwrite или skip")
        if on_error not in {"continue", "stop"}:
            raise ValueError("Режим ошибок очереди должен быть continue или stop")
        normalized = [str(Path(source).resolve()) for source in sources]
        if not normalized:
            raise ValueError("Для задания не выбраны исходные файлы")
        job = BatchJob(load_action(action), normalized, str(Path(destination).resolve()), suffix, conflict, on_error)
        with self._lock:
            self.jobs.append(job)
        return job

    def cancel(self) -> None:
        self._cancel.set()

    def remove_finished(self) -> int:
        with self._lock:
            before = len(self.jobs)
            self.jobs = [job for job in self.jobs if job.state in {"pending", "running"}]
            return before - len(self.jobs)

    def run_all(self, progress: Callable[[BatchJob, BatchItem], None] | None = None) -> list[BatchJob]:
        self._cancel.clear()
        with self._lock:
            pending = [job for job in self.jobs if job.state == "pending"]
        for job in pending:
            if self._cancel.is_set():
                job.state = "cancelled"
                break
            self._run_job(job, progress)
        return pending

    def _run_job(self, job: BatchJob, progress: Callable[[BatchJob, BatchItem], None] | None) -> None:
        output = Path(job.destination)
        output.mkdir(parents=True, exist_ok=True)
        job.state = "running"
        job.started_at = time.time()
        for item in job.items:
            if self._cancel.is_set():
                item.state = "cancelled"
                job.state = "cancelled"
                break
            item.state = "running"
            if progress:
                progress(job, item)
            try:
                source = Path(item.source)
                target = self._target_path(output, source.stem, job.suffix, job.conflict)
                if target is None:
                    item.state = "skipped"
                    item.message = "Файл результата уже существует"
                    continue
                document = Document.from_image(source)
                report = self.runner.run_with_report(
                    document,
                    job.action,
                    context={"source_extension": source.suffix.lower(), "source_name": source.name},
                    cancelled=self._cancel.is_set,
                )
                if self._cancel.is_set():
                    item.state = "cancelled"
                    job.state = "cancelled"
                    break
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp{target.suffix}")
                document.export_flat(temporary)
                temporary.replace(target)
                item.target = str(target)
                item.executed_steps = report.executed
                item.state = "completed"
                item.message = report.stop_message
            except SkipBatchFile as exc:
                item.state = "skipped"
                item.message = str(exc)
            except Exception as exc:
                item.state = "failed"
                item.message = str(exc)
                if job.on_error == "stop":
                    job.state = "failed"
                    if progress:
                        progress(job, item)
                    break
            finally:
                if progress:
                    progress(job, item)
        if job.state == "running":
            job.state = "completed_with_errors" if job.error_count else "completed"
        job.finished_at = time.time()

    @staticmethod
    def _target_path(folder: Path, stem: str, suffix: str, conflict: str) -> Path | None:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        target = folder / f"{stem}{suffix}"
        if not target.exists() or conflict == "overwrite":
            return target
        if conflict == "skip":
            return None
        number = 2
        while True:
            candidate = folder / f"{stem}_{number}{suffix}"
            if not candidate.exists():
                return candidate
            number += 1

    def save(self, path: str | Path) -> None:
        with self._lock:
            payload = {"format": "UZYRO batch queue v1", "jobs": [asdict(job) for job in self.jobs]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        supported_formats = {
            "UZYRO batch queue v1",
            f"{'Photo' + 'Redactor'} batch queue v1",
        }
        if payload.get("format") not in supported_formats:
            raise ValueError("Неподдерживаемый формат очереди")
        jobs: list[BatchJob] = []
        for raw in payload.get("jobs", []):
            raw = dict(raw)
            raw["items"] = [BatchItem(**item) for item in raw.get("items", [])]
            job = BatchJob(**raw)
            if job.state == "running":
                job.state = "pending"
            if job.state not in JOB_STATES:
                raise ValueError(f"Неизвестное состояние задания: {job.state}")
            jobs.append(job)
        with self._lock:
            self.jobs = jobs
        return len(jobs)


__all__ = ["BatchItem", "BatchJob", "BatchQueue"]

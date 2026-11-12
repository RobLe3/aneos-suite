"""Append-only local Sentry report archive, not a durable job or case workflow.

POSIX local-filesystem atomic hard links publish fsynced complete snapshots.
An existing report is verified, never overwritten. No destructive recovery.
"""
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from ..data.sources.sentry import verify_sentry_report


class EvidenceArchive:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('ANEOS_EVIDENCE_ARCHIVE', 'neo_data/evidence')).expanduser()

    def _root(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError('Unsafe evidence archive directory')
        return self.directory

    def get(self, report_id):
        if not isinstance(report_id, str) or not re.fullmatch('[0-9a-f]{64}', report_id):
            raise ValueError('Invalid evidence report ID')
        path = self._root() / (report_id + '.json')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'r', encoding='utf-8') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError('Unsafe evidence archive entry')
            report = json.load(handle)
        if not verify_sentry_report(report) or report['report_sha256'] != report_id:
            raise ValueError('Corrupt or modified evidence archive entry')
        return report

    def put(self, report):
        if not verify_sentry_report(report):
            raise ValueError('Only replay-verified Sentry evidence may be archived')
        root = self._root()
        final = root / (report['report_sha256'] + '.json')
        payload = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=root)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, final)  # Atomic, refuses an existing destination.
            except FileExistsError:
                if self.get(report['report_sha256']) != report:
                    raise ValueError('Conflicting evidence archive entry')
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            # Validate the actual published bytes before returning confirmation.
            if self.get(report['report_sha256']) != report:
                raise ValueError('Evidence archive verification failed')
            return final
        finally:
            Path(temporary).unlink(missing_ok=True)  # Only this call's staging file.

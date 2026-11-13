# Install an isolated aNEOS beta

Use the published **v2.2.0b2 engineering beta** for initial testing. It is not a
scientifically calibrated detector or a production warning service. The development
checkout includes additional work that the tag does not contain.

## Requirements

Use Python **3.11**, Git and network access to install dependencies. CI uses Python
3.11; this guide does not establish support for every Python/OS combination or a
minimum memory/performance guarantee. The shell commands below use macOS/Linux
syntax. Windows installation has not been requalified in this documentation pass.

Create a new checkout rather than updating an existing research working tree:

```bash
git clone --branch v2.2.0b2 https://github.com/RobLe3/aneos-suite.git aneos-suite-beta
cd aneos-suite-beta
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

Run subsequent commands from this checkout with the environment activated. The
core profile includes numerical, astronomy, visualization and API dependencies.
The optional full ML profile is not required for the introductory workflow and
does not enable a validated artificial-origin model.

## Keep test data separate

```bash
mkdir -p beta-runtime
export ANEOS_DATABASE_URL="sqlite:///$PWD/beta-runtime/aneos-beta.sqlite"
```

This selects a new **file-backed SQLite** database. Set the variable in each new
shell before querying. The commands neither migrate nor back up an existing
database. Do not point them at a live investigation or assume Docker's default
PostgreSQL service is a qualified replacement for this evidence-store workflow.
Database receipts, source payloads, exports and caches can contain research data;
review them before sharing or deleting anything.

## Check the installation without polling

```bash
python aneos.py --version
python aneos.py help
python aneos.py analyze --help
```

The version should be `aNEOS 2.2.0b2 (beta)`. Help commands do not query astronomical
providers. They establish that the selected entry point loads, not that every
provider is reachable or every scientific model is installed.

Continue with the [quick start](quick-start.md). For tag-specific limits, read
[the beta release guide](../releases/2.2.0b2.md).

## If something fails

- **Wrong interpreter or missing import:** check `python --version` and reinstall
  the core requirements in the activated environment. Record the failing package
  and error rather than installing an unrelated global package.
- **Wrong directory:** run from the folder containing `aneos.py` and `aneos_core/`.
- **Database error:** verify the configured path and parent directory permissions.
  Fresh acquisition needs a writable file-backed SQLite database. Historical
  read-only inspection is a different operation; do not remove read-only protections
  from research originals just to run a test.
- **Provider unavailable:** installation can be valid while a source is unavailable.
  Keep the recorded source outcome; do not replace missing measurements with zeros.
- **Output already exists:** choose a new filename. Report creation refuses to
  overwrite existing work; this is not a complete backup or recovery guarantee.

Authentication and deployment hardening remain separate work. Start development
API servers on explicit loopback only; do not expose them as a public service.

# ...existing code...
#!/usr/bin/env bash
set -euo pipefail

# Install requirements for the Dash app in this workspace.
# Usage: ./scripts/install_requirements.sh

REQ_FILE="$(dirname "$0")/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found. Please install Python 3."
  exit 1
fi

PYTHON=python3

# ensure pip is available
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "pip for python3 not found — attempting to install python3-pip via apt (requires sudo)."
  sudo apt update
  sudo apt install -y python3-pip
fi

# install from requirements.txt
echo "Installing python packages from $REQ_FILE..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --no-cache-dir -r "$REQ_FILE"

echo "Done. You can run the app with:"
echo '  $ python3 /workspaces/ORKs-network/app.py'

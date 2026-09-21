#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [ssh-host-or-config-alias]" >&2
    exit 2
fi
DEPLOY_HOST="${1:-${DEPLOY_HOST:-172.235.205.12}}"
DEPLOY_USER="${DEPLOY_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/PyBrosAI}"
SSH_PORT="${SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"
DEPLOY_RETRIES="${DEPLOY_RETRIES:-5}"
RETRY_DELAY="${RETRY_DELAY:-10}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"
SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-300}"

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required on the local machine." >&2
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    echo "ssh is required on the local machine." >&2
    exit 1
fi
SSH_ARGS=(
    -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=3
)
if [[ -n "${SSH_PORT}" ]]; then
    SSH_ARGS+=(-p "${SSH_PORT}")
fi
if [[ -n "${SSH_KEY}" ]]; then
    SSH_ARGS+=(-i "${SSH_KEY}")
fi

CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pybrosai-ssh.XXXXXX")"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pybrosai-deploy.XXXXXX")"
cleanup() {
    rm -rf "$CONTROL_DIR" "$STAGING_DIR"
}
trap cleanup EXIT
SSH_ARGS+=(
    -o ControlMaster=auto
    -o "ControlPersist=${SSH_CONTROL_PERSIST}"
    -o "ControlPath=${CONTROL_DIR}/control"
)

REMOTE="${DEPLOY_USER}@${DEPLOY_HOST}"
remote_dir_quoted="$(printf '%q' "$REMOTE_DIR")"
RSYNC_EXCLUDES=(
    "--exclude=.git/"
    "--exclude=.venv/"
    "--exclude=__pycache__/"
    "--exclude=checkpoints/"
    "--exclude=*.zip"
)
REMOTE_PRESERVE_EXCLUDES=(
    "--exclude=checkpoints/"
    "--exclude=roms/"
)
RSYNC_SSH=(ssh "${SSH_ARGS[@]}")
printf -v RSYNC_COMMAND '%q ' "${RSYNC_SSH[@]}"

ssh_command() {
    ssh "${SSH_ARGS[@]}" "$@"
}

run_remote() {
    ssh_command "$REMOTE" "$1"
}

retry() {
    local description="$1"
    shift
    local attempt=1

    while true; do
        if "$@"; then
            return 0
        fi
        if ((attempt >= DEPLOY_RETRIES)); then
            echo "${description} failed after ${DEPLOY_RETRIES} attempts." >&2
            return 1
        fi
        echo "${description} failed; retrying in ${RETRY_DELAY}s (${attempt}/${DEPLOY_RETRIES})..." >&2
        sleep "$RETRY_DELAY"
        attempt=$((attempt + 1))
    done
}

printf 'Staging %s\n' "$REPO_ROOT"
rsync -a --delete \
    "${RSYNC_EXCLUDES[@]}" \
    "$REPO_ROOT/" "$STAGING_DIR/"

printf 'Syncing staged code to %s:%s\n' "$REMOTE" "$REMOTE_DIR"
retry "Preparing remote directory" \
    run_remote "mkdir -p -- ${remote_dir_quoted}/roms ${remote_dir_quoted}/checkpoints"

retry "Syncing files" rsync -az --delete \
    -e "$RSYNC_COMMAND" \
    "${REMOTE_PRESERVE_EXCLUDES[@]}" \
    "$STAGING_DIR/" "$REMOTE:$REMOTE_DIR/"

rom_path_quoted="$(printf '%q' "$REMOTE_DIR/roms/Super Mario Bros. Deluxe.gbc")"
retry "Checking remote ROM" run_remote \
    "[ -f ${rom_path_quoted} ] || { echo 'Missing remote ROM: ${REMOTE_DIR}/roms/Super Mario Bros. Deluxe.gbc' >&2; exit 1; }"

echo "Building the Docker image..."
retry "Building Docker image" run_remote \
    "cd -- ${remote_dir_quoted} && docker compose build --no-cache"

echo "Starting the trainer..."
retry "Starting trainer" run_remote \
    "cd -- ${remote_dir_quoted} && docker compose up -d --force-recreate trainer && docker compose ps trainer"

if [[ "${FOLLOW_LOGS}" == "1" ]]; then
    run_remote "cd -- ${remote_dir_quoted} && docker compose logs -f --tail=50 trainer"
else
    echo "Deployment complete. Follow logs with:"
    echo "ssh ${SSH_ARGS[*]} ${REMOTE} 'cd ${REMOTE_DIR} && docker compose logs -f --tail=50 trainer'"
fi

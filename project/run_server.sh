#!/usr/bin/env bash
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
INTERNAL_DIR="$APP_DIR/_internal"
VENV_DIR="$APP_DIR/.venv"
REQ_FILE="$INTERNAL_DIR/requirements_server.txt"
SERVER_FILE="$INTERNAL_DIR/app_server.py"
PYTHON_BIN="python3"

cd "$APP_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Ошибка: python3 не найден."
    echo "Установи Python 3:"
    echo "sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-full"
    exit 1
fi

if [ ! -d "$INTERNAL_DIR" ]; then
    echo "Ошибка: не найдена папка $INTERNAL_DIR"
    exit 1
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "Ошибка: не найден $REQ_FILE"
    exit 1
fi

if [ ! -f "$SERVER_FILE" ]; then
    echo "Ошибка: не найден $SERVER_FILE"
    exit 1
fi

create_venv() {
    echo "Создаю виртуальное окружение .venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" || {
        echo "Не удалось создать venv."
        echo "На Debian установи:"
        echo "sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-full"
        exit 1
    }
}

ensure_pip_in_venv() {
    VENV_PYTHON="$VENV_DIR/bin/python"

    if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
        return 0
    fi

    echo "pip внутри .venv не найден. Пробую восстановить через ensurepip..."

    if "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
        return 0
    fi

    echo "ensurepip не сработал. Пересоздаю .venv..."

    rm -rf "$VENV_DIR"
    create_venv

    VENV_PYTHON="$VENV_DIR/bin/python"

    if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
        return 0
    fi

    if "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
        return 0
    fi

    echo "Не удалось добавить pip в .venv."
    echo "Проверь пакеты Debian:"
    echo "sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-full"
    exit 1
}

if [ ! -d "$VENV_DIR" ]; then
    create_venv
fi

ensure_pip_in_venv

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Обновляю pip..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

REQ_HASH_FILE="$VENV_DIR/.requirements_server.sha256"
CURRENT_REQ_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"

if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$CURRENT_REQ_HASH" ]; then
    echo "Устанавливаю/обновляю зависимости из _internal/requirements_server.txt..."
    "$VENV_PYTHON" -m pip install -r "$REQ_FILE"
    echo "$CURRENT_REQ_HASH" > "$REQ_HASH_FILE"
else
    echo "Зависимости уже установлены."
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
THREADS="${THREADS:-4}"

echo "Запускаю AGL Predictor Server..."
echo "URL: http://$HOST:$PORT"

exec "$VENV_PYTHON" "$SERVER_FILE" --host "$HOST" --port "$PORT" --threads "$THREADS"
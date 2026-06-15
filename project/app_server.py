# _internal/app_server.py

import os
import sys
from argparse import ArgumentParser

INTERNAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(INTERNAL_DIR)
SRC_DIR = os.path.join(INTERNAL_DIR, "src")

if not os.path.isdir(SRC_DIR):
    raise RuntimeError(f"Не найдена папка с исходниками: {SRC_DIR}")

# Важно:
# app_server.py лежит в _internal, а там же лежат Windows-библиотеки PyInstaller.
# На Linux нельзя давать Python импортировать пакеты из _internal.
# Поэтому убираем _internal из sys.path и добавляем только _internal/src.
internal_abs = os.path.abspath(INTERNAL_DIR)

clean_sys_path = []
for path in sys.path:
    if path:
        try:
            if os.path.abspath(path) == internal_abs:
                continue
        except Exception:
            pass

    clean_sys_path.append(path)

sys.path = clean_sys_path

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from waitress import serve
from app import app


def parse_args():
    parser = ArgumentParser(description="AGL Predictor Server")

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Адрес сервера"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Порт сервера"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Количество потоков Waitress"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"AGL Predictor server started: http://{args.host}:{args.port}")

    serve(
        app,
        host=args.host,
        port=args.port,
        threads=args.threads
    )

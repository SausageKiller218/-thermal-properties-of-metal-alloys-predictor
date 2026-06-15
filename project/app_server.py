# app_server.py

from argparse import ArgumentParser
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

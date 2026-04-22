"""
TradingView webhook receiver.
Listens for HTTP POST alerts from TradingView Pine Script and queues them.

TradingView alert message format (JSON):
    {"symbol": "SOXL", "action": "buy", "price": 23.45, "comment": "EMA cross"}

Usage:
    server = TVWebhookServer(port=8080, secret="optional_token")
    server.start()                          # runs in background thread
    signal = server.get_signal(timeout=5)   # returns dict or None
    server.stop()
"""
import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

log = logging.getLogger("tv_webhook")


class TVWebhookServer:
    def __init__(self, port: int = 8080, secret: Optional[str] = None):
        self._port = port
        self._secret = secret
        self._queue: queue.Queue = queue.Queue()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        handler = self._make_handler()
        self._server = HTTPServer(("0.0.0.0", self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info(f"TradingView webhook listening on port {self._port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            log.info("TradingView webhook server stopped")

    def get_signal(self, timeout: float = 1.0) -> Optional[dict]:
        """Pop the next queued alert, or None if queue is empty."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _make_handler(self):
        secret = self._secret
        signal_queue = self._queue

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Optional secret token check via query string ?token=...
                if secret:
                    from urllib.parse import urlparse, parse_qs
                    params = parse_qs(urlparse(self.path).query)
                    token = params.get("token", [None])[0]
                    if token != secret:
                        self.send_response(401)
                        self.end_headers()
                        return

                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")

                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    # Accept plain-text alerts too: "SOXL buy 23.45"
                    parts = body.strip().split()
                    payload = {"raw": body, "symbol": parts[0] if parts else ""}

                signal_queue.put(payload)
                log.info(f"Alert received: {payload}")

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")

            def log_message(self, fmt, *args):
                pass  # suppress default HTTP logs

        return Handler

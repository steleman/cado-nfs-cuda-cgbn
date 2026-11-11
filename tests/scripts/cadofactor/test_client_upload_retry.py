#!/usr/bin/env python3

# Check that cado-nfs-client re-sends the full result files when an
# upload is retried after a failed attempt.

import sys
import os
import threading
import types
import importlib.machinery
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer

# the client locates its python modules relative to sys.argv[0]
sys.argv = sys.argv[1:]
# (the client has no .py suffix, hence the explicit loader; load_module()
# is gone from recent pythons)
loader = importlib.machinery.SourceFileLoader("cado_nfs_client", sys.argv[0])
client = importlib.util.module_from_spec(
    importlib.util.spec_from_loader("cado_nfs_client", loader))
loader.exec_module(client)

PAYLOAD = b"c40 results payload\n" * 64
bodies = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        bodies.append(self.rfile.read(n))
        # first attempt is refused, the retry succeeds
        self.send_response(500 if len(bodies) == 1 else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


httpd = HTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

wdir = os.environ.get("wdir", ".")
result = os.path.join(wdir, "result.txt")
with open(result, "wb") as f:
    f.write(PAYLOAD)

server = client.Server(0, f"http://127.0.0.1:{httpd.server_port}")
wu = types.SimpleNamespace(is_stale=lambda: False,
                           get_id=lambda: "fake_wu",
                           get_peer=lambda: server)
p = client.WorkunitProcessor.__new__(client.WorkunitProcessor)
p.workunit = wu
p.cleanup = lambda: None
p._answer = dict(files={"result.txt": open(result, "rb")},
                 data=dict(WUid="fake_wu", clientid="test"))

settings = dict(DOWNLOADRETRY="1", DOWNLOADRETRYMIN="0.01",
                POSTRESULTPATH="upload", SILENT_WAIT=False)
up = client.ResultUploader(settings, types.SimpleNamespace(servers=[server]))
up.schedule_upload(p)
up.process_uploads()
httpd.shutdown()

assert len(bodies) == 2, f"expected 2 POSTs, got {len(bodies)}"
for i, b in enumerate(bodies):
    assert PAYLOAD in b, f"POST #{i} lacks the result file contents"
assert up.backlog_size == 0
print("ok")

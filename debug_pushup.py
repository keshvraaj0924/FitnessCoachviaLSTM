import urllib.request
import os

path = r"D:\AerioneBharat\demo_pushup.mp4"
boundary = "----FormBoundary7MA4YWxkTrZu0gW"
with open(path, "rb") as f:
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="exercise"\r\n\r\n'
        f"pushup\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="demo_pushup.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request("http://localhost:8000/v1/analyze", data=body, method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
try:
    r = urllib.request.urlopen(req)
    print(r.read().decode()[:500])
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    print(e.read().decode()[:500])

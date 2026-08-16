"""Check all demo videos with explicit exercise."""
import urllib.request
import json
import os

url = "http://localhost:8000/v1/analyze"
base = r"D:\AerioneBharat"

exercises = {
    "demo_pushup": "pushup",
    "demo_squat": "squat",
    "demo_bicep_curl": "bicep_curl",
    "demo_jumping_jack": "jumping_jack",
}

for fname, ex_id in exercises.items():
    path = os.path.join(base, f"{fname}.mp4")
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    with open(path, "rb") as f:
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="exercise"\r\n\r\n'
            f"{ex_id}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="video"; filename="{fname}.mp4"\r\n'
            f"Content-Type: video/mp4\r\n\r\n"
        ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        resp = urllib.request.urlopen(req)
        d = json.loads(resp.read())
        reps = d.get('rep_count', -1)
        coach = d.get('coaching_feedback', {}).get('summary', 'N/A')[:70]
        print(f"{fname}: {reps} reps | coach: {coach}")
    except Exception as e:
        print(f"{fname}: ERROR - {e}")

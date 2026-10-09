import subprocess
import re
import time
import csv
import sys
from datetime import datetime

INTERVAL = 0.2  # 200 ms，短时峰值比 1 s 采样更可靠

def get_value(text, key):
    m = re.search(rf"{re.escape(key)}\s*:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None

def sample():
    r = subprocess.run(
        ["hm_smi", "-a"],
        capture_output=True,
        text=True
    )

    text = r.stdout

    total = get_value(text, "DDR_Memory_Total")
    free = get_value(text, "DDR_Memory_Free")
    util = get_value(text, "Average_Util")
    ipu_load = get_value(text, "IPU_Load")

    if total is None or free is None:
        return None

    used = total - free

    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "used_mb": used,
        "total_mb": total,
        "memory_percent": used / total * 100,
        "average_util": util,
        "ipu_load": ipu_load,
    }


if "--" not in sys.argv:
    print("Usage:")
    print("python3 monitor_hm.py -- <inference command>")
    sys.exit(1)

idx = sys.argv.index("--")
cmd = sys.argv[idx + 1:]

if not cmd:
    print("No inference command.")
    sys.exit(1)

# 推理前基线
baseline = sample()

if baseline is None:
    print("ERROR: hm_smi -a 没有读取到 DDR_Memory 信息")
    sys.exit(1)

baseline_mb = baseline["used_mb"]

print(f"Baseline memory : {baseline_mb:.2f} MB")
print("Starting:", " ".join(cmd))

process = subprocess.Popen(cmd)

samples = []

while process.poll() is None:
    start = time.time()

    s = sample()

    if s:
        s["delta_mb"] = s["used_mb"] - baseline_mb
        samples.append(s)

    elapsed = time.time() - start
    time.sleep(max(0, INTERVAL - elapsed))

return_code = process.wait()

if not samples:
    print("No valid samples.")
    sys.exit(return_code)

with open("hm_inference.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=samples[0].keys())
    writer.writeheader()
    writer.writerows(samples)

used = [x["used_mb"] for x in samples]
delta = [x["delta_mb"] for x in samples]
mem_pct = [x["memory_percent"] for x in samples]

utils = [
    x["average_util"]
    for x in samples
    if x["average_util"] is not None
]

loads = [
    x["ipu_load"]
    for x in samples
    if x["ipu_load"] is not None
]

print()
print("========== Inference Resource Summary ==========")
print(f"Samples                 : {len(samples)}")
print(f"Baseline memory         : {baseline_mb:.2f} MB")
print()
print(f"Peak memory             : {max(used):.2f} MB")
print(f"Average memory          : {sum(used)/len(used):.2f} MB")
print(f"Peak inference delta    : {max(delta):.2f} MB")
print(f"Average inference delta : {sum(delta)/len(delta):.2f} MB")
print()
print(f"Peak memory usage       : {max(mem_pct):.2f} %")
print(f"Average memory usage    : {sum(mem_pct)/len(mem_pct):.2f} %")

if utils:
    print()
    print(f"Peak compute utilization: {max(utils):.2f} %")
    print(f"Avg compute utilization : {sum(utils)/len(utils):.2f} %")

if loads:
    print(f"Peak IPU load           : {max(loads):.2f} %")
    print(f"Avg IPU load            : {sum(loads)/len(loads):.2f} %")

print()
print("Raw data: hm_inference.csv")
print("================================================")

#!/usr/bin/env python3
"""
PI0.5 模型纯推理性能 benchmark + 端到端
"""

import argparse
import os
import time

import numpy as np
import tcim_lite


def make_input(info, seed=0):
    rng = np.random.default_rng(seed)
    shape = list(info.shape)
    dtype = info.dtype
    if dtype in (np.int8, np.uint8):
        return np.zeros(shape, dtype=dtype)
    if dtype in (np.int32, np.int64):
        return rng.integers(0, 100, size=shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def bench_model(model_path, warmup=3, samples=10):
    name = os.path.basename(model_path)
    t0 = time.time()
    module = tcim_lite.runtime.load(model_path)
    load_ms = (time.time() - t0) * 1000

    input_names = [module.get_input_name(i) for i in range(module.get_num_inputs())]
    input_arrays = {}
    input_infos = []
    for i, nm in enumerate(input_names):
        info = module.get_input_info(nm)
        input_arrays[nm] = make_input(info, seed=i)
        input_infos.append((nm, list(info.shape), info.dtype.__name__))

    # 跳过 auto_profile 输出
    output_names = [module.get_output_name(i) for i in range(module.get_num_outputs())
                    if "auto_profile" not in module.get_output_name(i)]
    output_infos = []
    for on in output_names:
        info = module.get_output_info(on)
        output_infos.append((on, list(info.shape), info.dtype.__name__))

    for _ in range(warmup):
        for nm, arr in input_arrays.items():
            module.set_input(nm, arr)
        module.run(); module.sync()

    set_times, run_times = [], []
    for _ in range(samples):
        t0 = time.time()
        for nm, arr in input_arrays.items():
            module.set_input(nm, arr)
        set_times.append((time.time() - t0) * 1000)

        t0 = time.time()
        module.run(); module.sync()
        run_times.append((time.time() - t0) * 1000)

    return {
        "name": name,
        "load_ms": round(load_ms, 1),
        "set_input_ms": round(float(np.mean(set_times)), 3),
        "run_ms": round(float(np.mean(run_times)), 3),
        "run_min_ms": round(float(np.min(run_times)), 3),
        "run_max_ms": round(float(np.max(run_times)), 3),
        "n_inputs": len(input_names),
        "input_infos": input_infos,
        "output_infos": output_infos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--hmm-dir", default="/workspace/PI0.5/example/outputs/xh2")
    args = parser.parse_args()

    models = [
        "siglip", "gemma_2b_prefill", "gemma_expert_300m_decode",
        "action_in_proj", "action_out_proj", "time_mlp",
    ]

    results = []
    print(f"{'='*80}")
    print(f"  PI0.5 单段推理基准 (warmup={args.warmup}, samples={args.samples})")
    print(f"{'='*80}")
    for m in models:
        p = os.path.join(args.hmm_dir, f"{m}.hmm")
        print(f"\n  Benchmarking {m} ...")
        r = bench_model(p, args.warmup, args.samples)
        results.append(r)
        print(f"    load={r['load_ms']}ms | set_input={r['set_input_ms']}ms | "
              f"run={r['run_ms']}ms (min={r['run_min_ms']}, max={r['run_max_ms']}) | "
              f"inputs={r['n_inputs']}")
        for nm, shape, dtype in r["input_infos"]:
            print(f"      in  [{nm}] shape={shape} dtype={dtype}")
        for nm, shape, dtype in r["output_infos"]:
            print(f"      out [{nm}] shape={shape} dtype={dtype}")

    print(f"\n\n{'='*80}")
    print(f"{'Model':<30} {'load(ms)':>10} {'set_in(ms)':>11} {'run(ms)':>9} "
          f"{'min(ms)':>9} {'max(ms)':>9} {'inputs':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['load_ms']:>10.1f} {r['set_input_ms']:>11.3f} "
              f"{r['run_ms']:>9.3f} {r['run_min_ms']:>9.3f} {r['run_max_ms']:>9.3f} "
              f"{r['n_inputs']:>7}")


if __name__ == "__main__":
    main()

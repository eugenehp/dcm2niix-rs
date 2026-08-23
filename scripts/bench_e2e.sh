#!/usr/bin/env bash
# End-to-end conversion bench across dcm_qa* corpora (warmed medians).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_CPU="${BIN_CPU:-$ROOT/target/release/dcm2niix-cpu}"
BIN_GPU="${BIN_GPU:-$ROOT/target/release/dcm2niix-gpu}"
BIN_CPP="${BIN_CPP:-}"
WARM="${WARM:-2}"
RUNS="${RUNS:-5}"
SHARED="$(cd "$ROOT/.." && pwd)"
OUT="${TMPDIR:-/tmp}/dcm2niix-bench-$$"
mkdir -p "$OUT"

median() {
  python3 -c 'import sys; v=sorted(float(x) for x in sys.stdin if x.strip()); print(v[len(v)//2] if v else float("nan"))'
}

time_one() {
  local bin="$1" indir="$2" outdir="$3" flags="$4"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  python3 - "$bin" "$indir" "$outdir" "$flags" <<'PY'
import subprocess, sys, time, shlex
bin, indir, outdir, flags = sys.argv[1:5]
cmd = [bin, "-b", "n", "-z", "n"] + shlex.split(flags) + ["-o", outdir, indir]
t0 = time.perf_counter()
r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
dt = time.perf_counter() - t0
if r.returncode != 0:
    sys.stderr.write(r.stderr.decode("utf-8", "replace")[:500])
    sys.exit(r.returncode)
print(f"{dt:.6f}")
PY
}

bench_corpus() {
  local name="$1" indir="$2" flags="$3"
  if [[ ! -d "$indir" ]]; then
    echo "skip $name (missing $indir)"
    return 0
  fi
  echo "══ $name  ($indir)  flags='$flags'  warm=$WARM runs=$RUNS ══"
  printf "%-18s  %s\n" "binary" "median wall (s)"
  for label_bin in "rust-cpu:$BIN_CPU" "rust-gpu:$BIN_GPU"; do
    local label="${label_bin%%:*}"
    local bin="${label_bin#*:}"
    if [[ ! -x "$bin" ]]; then
      printf "%-18s  (missing — build release + cp to dcm2niix-cpu/gpu)\n" "$label"
      continue
    fi
    local i t med
    for i in $(seq 1 "$WARM"); do
      time_one "$bin" "$indir" "$OUT/w-${label}-$name" "$flags" >/dev/null
    done
    local times=()
    for i in $(seq 1 "$RUNS"); do
      t=$(time_one "$bin" "$indir" "$OUT/t-${label}-$name" "$flags")
      times+=("$t")
    done
    med=$(printf '%s\n' "${times[@]}" | median)
    printf "%-18s  %.4f   [%s]\n" "$label" "$med" "$(printf '%s ' "${times[@]}")"
  done
  if [[ -n "$BIN_CPP" && -x "$BIN_CPP" ]]; then
    local times=() i t med
    for i in $(seq 1 "$WARM"); do
      time_one "$BIN_CPP" "$indir" "$OUT/w-cpp-$name" "$flags" >/dev/null
    done
    for i in $(seq 1 "$RUNS"); do
      t=$(time_one "$BIN_CPP" "$indir" "$OUT/t-cpp-$name" "$flags")
      times+=("$t")
    done
    med=$(printf '%s\n' "${times[@]}" | median)
    printf "%-18s  %.4f   [%s]\n" "cpp" "$med" "$(printf '%s ' "${times[@]}")"
  fi
  echo
}

echo "dcm2niix-rs end-to-end bench"
echo "CPU=$BIN_CPU"
echo "GPU=$BIN_GPU"
echo "CPP=${BIN_CPP:-none}"
echo

bench_corpus "dcm_qa" "$SHARED/dcm_qa/In" "-f %p_%s"
bench_corpus "dcm_qa_nih" "$SHARED/dcm_qa_nih/In" "-f %p_%s"
bench_corpus "dcm_qa_uih" "$SHARED/dcm_qa_uih/In" "-f %p_%s_%t"

rm -rf "$OUT"

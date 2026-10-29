#!/usr/bin/env bash
# Serve Qwen3.8-27B GSQ-RCO IQ3_XXS with the settings measured on the RTX 5060 Ti 16 GB.
#
#   scripts/qwen38/serve.sh [options] [-- more ninfer-serve arguments]
#
# The server speaks the OpenAI Chat Completions/Responses, Anthropic Messages and llama.cpp APIs at
# http://HOST:PORT/v1 and serves a WebUI at http://HOST:PORT/ (when the build has one). Ctrl+C stops
# it. Every choice below was measured on an RTX 5060 Ti 16 GB under WSL2 (docs: wiki/Serving.md).
#
# Options:
#   --context N         tokens per request. Default 262144 with up to 3 requests, 196608 with 4-8,
#                       131072 with 9-12, 98304 above; with --prefix-cache alt, 196608 (131072 at 5+)
#   --concurrency N     requests served at once, 1 (default) to 32. Above 1 they share one KV pool
#                       sized to the free GPU memory, and a prompt is processed while others generate
#   --spec mtp|none     speculative decoding with the model's MTP head (default) or none
#   --draft-tokens N    MTP drafts per round: default 4 with one request, 3 with more
#   --copy-tokens N     tokens copied per round from earlier text the reply repeats (n-gram drafts),
#                       0 disables. Default 31 with one request; 15 up to 16 requests, 7 up to 24,
#                       0 above. Above 8 requests only 0, 7 or 15
#   --prefix-cache catalog|alt
#                       catalog (default) keeps finished conversations for reuse; alt also lets a
#                       request wait for and reuse a concurrent request's shared prompt prefix
#                       (needs --concurrency 2-8)
#   --lease-growth      with --concurrency 2+: reserve prompt + 4096 tokens and grow the reservation,
#                       so long-context requests can run together
#   --ram-cache-gib N   pinned system RAM, 2-40 GiB, that keeps finished conversations' KV and model
#                       state so a returning conversation skips its history. Default: 22, or 45% of
#                       this system's RAM when that is less (at least 2); 8 with --prefix-cache alt
#   --temperature X     default sampling temperature 0-2 (0, greedy, by default); requests override
#   --thinking          think by default (off by default); requests override
#   --port N            default 8080
#   --local-only        listen on 127.0.0.1 only (default: every interface)
#   --model PATH        default models/qwen38-27b-gsq-iq3xxs.ninfer, or $NINFER_MODEL
#   --server PATH       default build-qwen38/apps/ninfer-serve, or $NINFER_SERVER
#   --print             print the command instead of running it
#
# The 262144 context fits a 16 GB card with about 270 MiB to spare. Programs that take more GPU
# memory after startup (games, other GPU programs, a browser on a Windows desktop) make a WDDM
# driver page GPU memory to system RAM, which can halve speed: close them or pass a smaller
# --context if speed drops.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${NINFER_MODEL:-$root/models/qwen38-27b-gsq-iq3xxs.ninfer}"
server="${NINFER_SERVER:-$root/build-qwen38/apps/ninfer-serve}"
context='' concurrency=1 spec=mtp draft='' copy='' port=8080 host=0.0.0.0 temperature=0
thinking=0 lease=0 prefix=catalog ram='' print=0
extra=()

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
fail() { printf 'serve.sh: %s\n' "$*" >&2; exit 2; }
integer() { [[ $2 =~ ^[0-9]+$ ]] && (( $2 >= $3 && $2 <= $4 )) || fail "$1 takes an integer from $3 to $4"; }
value() { (( $# >= 2 )) || fail "$1 needs a value"; }

while (( $# )); do
  case "$1" in
    --context) value "$@"; integer "$1" "$2" 1 262144; context=$2; shift 2 ;;
    --concurrency) value "$@"; integer "$1" "$2" 1 32; concurrency=$2; shift 2 ;;
    --spec) value "$@"; [[ $2 == mtp || $2 == none ]] || fail '--spec takes mtp or none'; spec=$2; shift 2 ;;
    --draft-tokens) value "$@"; integer "$1" "$2" 1 15; draft=$2; shift 2 ;;
    --copy-tokens) value "$@"; integer "$1" "$2" 0 63; copy=$2; shift 2 ;;
    --prefix-cache) value "$@"; [[ $2 == catalog || $2 == alt ]] || fail '--prefix-cache takes catalog or alt'; prefix=$2; shift 2 ;;
    --lease-growth) lease=1; shift ;;
    --ram-cache-gib) value "$@"; integer "$1" "$2" 2 40; ram=$2; shift 2 ;;
    --temperature) value "$@"
      awk -v t="$2" 'BEGIN { exit !(t ~ /^[0-9]+(\.[0-9]+)?$/ && t + 0 <= 2) }' || fail '--temperature takes a number from 0 to 2'
      temperature=$2; shift 2 ;;
    --thinking) thinking=1; shift ;;
    --port) value "$@"; integer "$1" "$2" 1 65535; port=$2; shift 2 ;;
    --local-only) host=127.0.0.1; shift ;;
    --model) value "$@"; model=$2; shift 2 ;;
    --server) value "$@"; server=$2; shift 2 ;;
    --print) print=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra=("$@"); break ;;
    *) fail "unknown option $1 (see --help)" ;;
  esac
done

# The largest context per request whose automatic pool still holds one such request, measured with
# each band's default copy window.
maximum_context() {
  if (( $1 <= 3 )); then echo 262144; elif (( $1 <= 8 )); then echo 196608
  elif (( $1 <= 12 )); then echo 131072; else echo 98304; fi
}

# CUDA Graph memory with MTP, measured: 18 MiB at one lane, ~16 MiB per further lane, plus ~116 MiB
# once there are five (the 15-token copy window's graphs). Above eight, measured per copy window at
# 8/12/16/24/32 requests, linear between points, plus 32 MiB.
graph_allowance() {
  local lanes=$1 window=$2
  if (( lanes <= 8 )); then
    local used=$(( 16 * lanes + (lanes >= 5 ? 108 : 8) ))
    echo $(( used > 32 ? used : 32 ))
    return
  fi
  local points
  case $window in
    15) points='8:228 12:538 16:846' ;;
    7) points='8:228 16:268 24:596 32:928' ;;
    *) points='8:228 16:86 24:226 32:268' ;;
  esac
  awk -v n="$lanes" -v p="$points" 'BEGIN {
    k = split(p, q, " ")
    for (i = 2; i <= k; ++i) {
      split(q[i - 1], a, ":"); split(q[i], b, ":")
      if (n <= b[1] || i == k) {
        u = a[2] + (b[2] - a[2]) * (n - a[1]) / (b[1] - a[1])
        c = int(u); if (c < u) c += 1
        print c + 32; exit
      }
    }
  }'
}

alt=0
[[ $prefix == alt ]] && alt=1
if [[ -z $context ]]; then
  if (( alt )); then context=$(( concurrency >= 5 ? 131072 : 196608 )); else context=$(maximum_context "$concurrency"); fi
fi
if [[ -z $ram ]]; then
  if (( alt )); then ram=8; else
    system_gib=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
    ram=$(( system_gib * 45 / 100 ))
    (( ram > 22 )) && ram=22
    (( ram < 2 )) && ram=2
  fi
fi
[[ -n $draft ]] || draft=$(( concurrency >= 2 ? 3 : 4 ))
if [[ -z $copy ]]; then
  if (( concurrency >= 2 )); then
    copy=$(( concurrency > 24 ? 0 : (concurrency > 16 ? 7 : 15) ))
  else
    copy=31
  fi
fi

limit=$(maximum_context "$concurrency")
(( context <= limit )) || fail "--concurrency $concurrency fits a context of at most $limit tokens per request on a 16 GB card; pass --context $limit or less, or fewer requests"
(( ! lease || concurrency > 1 )) || fail '--lease-growth requires --concurrency 2 or more'
(( copy <= 15 || concurrency == 1 )) || fail "--copy-tokens $copy needs --concurrency 1; the engine copies at most 15 tokens per round above one request"
if (( concurrency > 8 )) && [[ $copy != 0 && $copy != 7 && $copy != 15 ]]; then
  fail "--copy-tokens $copy with --concurrency $concurrency: above eight requests pass 0, 7 or 15"
fi
(( ! (concurrency > 16 && copy == 15) )) || fail '--copy-tokens 15 fits at most 16 requests; pass --copy-tokens 7 or 0'
(( ! (concurrency > 24 && copy == 7) )) || fail '--copy-tokens 7 leaves too small a pool above 24 requests; pass --copy-tokens 0'
if (( alt )); then
  (( concurrency > 1 )) || fail '--prefix-cache alt shares work between concurrent requests; pass --concurrency 2 or more'
  (( concurrency <= 8 )) || fail '--prefix-cache alt is measured up to eight requests; pass --concurrency 8 or less'
  (( ! (concurrency <= 4 && context > 196608) )) || fail '--prefix-cache alt fits a context of at most 196608 tokens; pass --context 196608 or less'
  (( ! (concurrency >= 5 && context > 131072) )) || fail "--prefix-cache alt with --concurrency $concurrency fits a context of at most 131072 tokens"
fi

args=("$model" --host "$host" --port "$port" --model-id qwen3.8-27b
  --max-context "$context" --max-concurrency "$concurrency" --kv-dtype rk4v4 --gdn-state-fp16)
# Progress anchors (a cancelled long prefill resumes from the deepest) every 27648 tokens: a multiple
# of every prefill chunk used here (1152, 2304, 3456, 4608), so they split no chunk.
(( alt )) || args+=(--progress-anchor-tokens 27648)
# Above 196608, or with concurrent requests sharing a pool sized to the free memory: token embedding
# in host memory, no extra device checkpoint states, and the measured CUDA Graph allowance (the
# engine's own default reserves 13-15x that).
if (( context > 196608 || concurrency > 1 )); then
  args+=(--embedding-host --cuda-graph-allowance-mib "$(graph_allowance "$concurrency" "$copy")")
  if (( alt )); then args+=(--use-alt-prefix-caching); else args+=(--device-state-slots 0); fi
fi
# One request above 196608: the automatic chunk (4608) leaves 136 MiB; 2304 leaves about 300 MiB,
# enough for the display and a browser, and prefills a 160K prompt 2% slower.
(( concurrency == 1 && context > 196608 )) && args+=(--prefill-chunk 2304)
if (( concurrency > 1 )); then
  # The widest chunk measured to fit at 160 MiB headroom.
  if (( context <= 131072 )); then chunk=3456
  elif (( context <= 196608 || concurrency == 2 )); then chunk=2304
  else chunk=1152; fi
  args+=(--concurrent-prefill --kv-capacity auto --kv-headroom-mib 160 --prefill-chunk "$chunk")
  (( lease )) && args+=(--kv-lease-growth)
fi
# Host RAM tier. Catalog: up to 64 conversations, model states (75 MiB each with FP16 GDN state) take
# up to half the budget (at most 128), KV the rest.
ram_mib=$(( ram * 1024 ))
if (( alt )); then
  args+=(--host-cache-mib "$ram_mib")
else
  state_bytes=78456832
  states=$(( ram_mib * 1048576 / 2 / state_bytes ))
  (( states > 128 )) && states=128
  kv_mib=$(( ram_mib - (states * state_bytes + 1048575) / 1048576 ))
  args+=(--max-private-continuations 64 --host-state-slots "$states" --host-kv-mib "$kv_mib")
fi
[[ $spec == none ]] || args+=(--spec "$spec" --draft-tokens "$draft" --ngram-draft-tokens "$copy")
args+=(--temperature "$temperature")
(( thinking )) || args+=(--no-thinking)
args+=("${extra[@]}")

if (( print )); then
  printf '%q ' "$server" "${args[@]}"
  printf '\n'
  exit 0
fi
[[ -x $server ]] || fail "server not found: $server (build it with scripts/qwen38/build.sh, or pass --server)"
[[ -f $model ]] || fail "model not found: $model (prepare it with scripts/qwen38/prepare-model.sh, or pass --model)"
exec "$server" "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

##
## SOURCE: https://github.com/exo-explore/exo/blob/main/configure_mlx.sh
##
## # POV benchmark on M4 Max 128GB RAM with meta-llama/Meta-Llama-3-8B-Instruct
##
## INPUT 15 tokens, OUTPUT 101 tokens.
##
## ttft: time to fist token, ts: tokens per second
##
## ## baseline:
## - run #1 - ttft 0.32s, ts 20.30 t/s
## - run #2: ttft 0.31s, ts 20.43 t/s
## - run #3: ttft 0.28s, ts 20.56 t/s
## - run #4: ttft 0.28s, ts 20.54 t/s
## - run #5: ttft 0.28s, ts 20.58 t/s
##
## ## with this script:
## - run #1: ttft 0.19s, ts 20.86 t/s
## - run #2: ttft 0.18s, ts 20.80 t/s
## - run #3: ttft 0.18s, ts 20.92 t/s
## - run #4: ttft 0.18s, ts 20.88 t/s
## - run #5: ttft 0.17s, ts 20.87 t/s
##

# Get the total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))

# Calculate 80% and TOTAL_MEM_GB-5GB in MB
EIGHTY_PERCENT=$(($TOTAL_MEM_MB * 80 / 100))
MINUS_5GB=$((($TOTAL_MEM_MB - 5120)))

# Calculate 70% and TOTAL_MEM_GB-8GB in MB
SEVENTY_PERCENT=$(($TOTAL_MEM_MB * 70 / 100))
MINUS_8GB=$((($TOTAL_MEM_MB - 8192)))

# Set WIRED_LIMIT_MB to higher value
if [ $EIGHTY_PERCENT -gt $MINUS_5GB ]; then
  WIRED_LIMIT_MB=$EIGHTY_PERCENT
else
  WIRED_LIMIT_MB=$MINUS_5GB
fi

# Set WIRED_LWM_MB to higher value
if [ $SEVENTY_PERCENT -gt $MINUS_8GB ]; then
  WIRED_LWM_MB=$SEVENTY_PERCENT
else
  WIRED_LWM_MB=$MINUS_8GB
fi

# Display the calculated values
echo "Total memory: $TOTAL_MEM_MB MB"
echo "Maximum limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB"
echo "Lower bound (iogpu.wired_lwm_mb): $WIRED_LWM_MB MB"

# Apply the values with sysctl, but check if we're already root
if [ "$EUID" -eq 0 ]; then
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
else
  # Try without sudo first, fall back to sudo if needed
  sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB
  sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB 2>/dev/null || \
    sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB
fi

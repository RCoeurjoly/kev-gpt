#!/usr/bin/env bash
set -o pipefail

log_file=$1
shift

"$@" 2>&1 | tee "$log_file"

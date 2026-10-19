#!/bin/bash
# s50_series.sh: interleaved S50 runs, A (PR #91 head 4c735f8) and B (e190a35), via s50_run.sh.
# Usage: SER=<name> bash s50_series.sh <label=binary> ...
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
for arm in "$@"; do
  label=${arm%%=*}; bin=${arm#*=}
  OUT=$HERE/raw/s50/$SER/$label BIN=$bin LABEL=$label bash "$HERE/s50_run.sh"
  sleep 5
done
echo SERIES_DONE

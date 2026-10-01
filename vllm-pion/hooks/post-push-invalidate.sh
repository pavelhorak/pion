#!/bin/bash
# Git post-push hook for Pion Serve cache invalidation.
#
# Sends changed files from the latest push to Pion Serve's invalidation
# endpoint, ensuring the KV cache doesn't serve completions based on
# outdated code.
#
# Install:
#   cp hooks/post-push-invalidate.sh .git/hooks/post-push
#   chmod +x .git/hooks/post-push
#
# Or configure as a GitHub webhook:
#   URL: https://your-pion-serve:8000/v1/invalidate
#   Content type: application/json
#   Events: Push
#
# Environment:
#   PION_SERVE_URL  — Pion Serve base URL (default: http://localhost:8000)

set -euo pipefail

PION_SERVE_URL="${PION_SERVE_URL:-http://localhost:8000}"

# Get changed files from the latest commit
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
    exit 0
fi

# Build JSON payload
FILES_JSON=$(echo "$CHANGED_FILES" | jq -R -s 'split("\n") | map(select(length > 0))')
PAYLOAD="{\"files\": $FILES_JSON}"

# Send to Pion Serve
RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "${PION_SERVE_URL}/v1/invalidate/files" \
    --connect-timeout 2 \
    --max-time 5 \
    2>/dev/null || echo '{"error": "connection failed"}')

# Parse result
INVALIDATED=$(echo "$RESPONSE" | jq -r '.invalidated // "?"' 2>/dev/null || echo "?")

if [ "$INVALIDATED" != "?" ] && [ "$INVALIDATED" != "0" ]; then
    echo "[pion-serve] Invalidated $INVALIDATED cache entries for $(echo "$CHANGED_FILES" | wc -l | tr -d ' ') changed files"
fi

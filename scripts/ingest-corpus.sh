#!/usr/bin/env bash
# Uploads every document in corpus/ and waits until each is indexed.
#
# Ingestion is asynchronous, so the upload returning 202 means "queued", not
# "searchable". Anything that queries straight after uploading — a demo, the
# evaluation harness, a test — has to wait for status to reach `ready`, which is
# what this script exists to do.
set -euo pipefail

API="${API:-http://localhost:8081}"
CORPUS="${CORPUS:-corpus}"
TIMEOUT="${TIMEOUT:-300}"

ids=()
for file in "$CORPUS"/*.md; do
  title=$(head -n 20 "$file" | grep -m1 '^# ' | sed 's/^# //' || basename "$file")
  response=$(curl -sS -X POST "$API/v1/documents" \
    -H "Content-Type: text/markdown" \
    -H "X-Title: $title" \
    -H "X-Source-URI: corpus/$(basename "$file")" \
    --data-binary "@$file")
  id=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
  dup=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["duplicate"])')
  printf '  queued %-28s %s%s\n' "$(basename "$file")" "$id" "$([ "$dup" = True ] && echo '  (already present)' || true)"
  ids+=("$id")
done

echo "waiting for ${#ids[@]} documents to be indexed..."
deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  pending=0
  for id in "${ids[@]}"; do
    status=$(curl -sS "$API/v1/documents/$id" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    case "$status" in
      ready) ;;
      failed)
        echo "document $id failed to ingest:"
        curl -sS "$API/v1/documents/$id" | python3 -m json.tool
        exit 1 ;;
      *) pending=$((pending + 1)) ;;
    esac
  done
  [ "$pending" -eq 0 ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out with $pending documents still pending" >&2
    exit 1
  fi
  sleep 3
done

curl -sS "$API/v1/stats" | python3 -c '
import json, sys
c = json.load(sys.stdin)["corpus"]
print("indexed {}/{} documents, {} chunks".format(
    c["ready_documents"], c["documents"], c["chunks"]))'

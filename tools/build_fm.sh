#!/usr/bin/env bash
# Build the FoundationModels backend CLI (macOS 26+, Apple Silicon).
set -euo pipefail
cd "$(dirname "$0")"
swiftc -O -parse-as-library fm-generate.swift -o fm-generate
echo "✓ built tools/fm-generate — test: ./fm-generate --prompt 'hi'"

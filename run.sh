#!/usr/bin/env bash
fuser -k 8080/tcp 2>/dev/null
sleep 0.5
exec uv run python main.py

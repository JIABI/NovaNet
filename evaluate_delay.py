#!/usr/bin/env python3
"""Compatibility entry point for trace-driven packet latency evaluation."""

from experiments.latency_from_traces import main


if __name__ == "__main__":
    raise SystemExit(main())


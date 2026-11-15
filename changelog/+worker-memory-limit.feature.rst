Add ``--worker-max-virtual-memory-gib`` and the ``worker_max_virtual_memory_gib``
configuration option to cap each worker's virtual address space on Linux and
macOS 12 or later using ``RLIMIT_AS``. Disabled by default, this lets runaway
allocations fail in the offending worker while preserving stricter existing limits.

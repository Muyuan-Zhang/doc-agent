"""Single source of truth for M4 Redis key formats.

Both the router (reads) and the consumer (writes) must use the same
key-generation logic or jobs become permanently unresolvable.
"""


def job_key(job_id: str) -> str:
    """Hash-tagged Redis key for a job's status hash.

    Hash tag on job_id ensures all related keys for the same job
    route to the same cluster slot.
    """
    return f"{{agent:job:{job_id}}}"

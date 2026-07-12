"""
Pipeline error taxonomy.

PipelineError carries a ready-to-display Hebrew message separately from the
technical detail. Raise it at the point of failure — where the context is
known — instead of string-matching exception text in the processor.
"""


class PipelineError(Exception):
    """Raised anywhere in the pipeline with a user-facing Hebrew message.

    user_message — shown to the user in the UI (stored in tasks.error).
    detail       — technical cause for debugging (stored in tasks.error_detail).
    """

    def __init__(self, user_message: str, *, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail

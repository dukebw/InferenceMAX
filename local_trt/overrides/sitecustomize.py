import os


def _maybe_prepend_trtllm_path() -> None:
    override_path = os.environ.get(
        "TRTLLM_OVERRIDE_PATH", "/workspace/tensorRT-LLM/tensorrt_llm")
    if not os.path.isdir(override_path):
        return
    try:
        import tensorrt_llm as tllm
    except Exception:
        return
    if override_path not in tllm.__path__:
        tllm.__path__.insert(0, override_path)


_maybe_prepend_trtllm_path()

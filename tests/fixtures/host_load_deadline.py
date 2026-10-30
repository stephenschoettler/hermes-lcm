"""Host loader snapshot for AST extraction with injected test dependencies."""
from __future__ import annotations


def run_with_load_deadline(plugin_key: str, ctx: PluginContext, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` (a plugin's import + ``register()``) under the per-plugin deadline.

    The worker inherits the caller's context (the Hermes-home override is a ContextVar). On timeout the
    worker is abandoned as a daemon, ``ctx`` is marked so any registration it still attempts is ignored,
    and :class:`PluginLoadTimeout` is raised on the calling thread so the usual failure path records the
    reason and disposes whatever was registered before the hang.
    """
    # Re-entrant load on our own deadline worker: run inline; the outer deadline covers it.
    outer_scope = _IN_PLUGIN_LOAD.get()
    if outer_scope is not None:
        outer_scope.append(ctx)
        try:
            return fn()
        finally:
            outer_scope.remove(ctx)
    timeout = _resolve_plugin_load_timeout()
    if timeout <= 0:
        return fn()
    _reserve_abandoned_loader_slot()
    outcome: list[Any] = []
    failure: list[BaseException] = []

    scope = [ctx]

    def _worker() -> None:
        token = _IN_PLUGIN_LOAD.set(scope)
        try:
            outcome.append(fn())
        except BaseException as exc:  # re-raised on the loading thread, KeyboardInterrupt included
            failure.append(exc)
        finally:
            _IN_PLUGIN_LOAD.reset(token)

    worker = threading.Thread(
        target=contextvars.copy_context().run, args=(_worker,), name=f"{_LOADER_THREAD_PREFIX}{plugin_key}", daemon=True,
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        for loaded_ctx in scope:
            loaded_ctx._abandon_load()
        with _ABANDONED_LOADERS_LOCK:
            _ABANDONED_LOADERS.append(worker)
        raise PluginLoadTimeout(f"load timed out after {timeout:g}s (import + register() never returned)")
    if failure:
        raise failure[0]
    return outcome[0]

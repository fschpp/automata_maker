"""Ejecuta trabajos largos en un hilo aparte quedándose solo con el último pedido."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class LatestJobRunner:
    """``submit(item)`` corre ``work(item)`` en un hilo. Si llegan pedidos mientras corre uno, solo
    se ejecuta el más reciente al terminar. ``on_done(item, result, error)`` se llama desde el hilo
    de trabajo, y solo para resultados que no quedaron obsoletos (la UI debe pasarlo a su hilo
    principal, p. ej. con ``GLib.idle_add``)."""

    def __init__(
        self,
        work: Callable[[Any], Any],
        on_done: Callable[[Any, Any, BaseException | None], None],
    ) -> None:
        self._work = work
        self._on_done = on_done
        self._lock = threading.Lock()
        self._pending: tuple[Any] | None = None
        self._running = False
        self._idle = threading.Event()
        self._idle.set()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def submit(self, item: Any) -> None:
        with self._lock:
            self._pending = (item,)
            if self._running:
                return
            self._running = True
            self._idle.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def _loop(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._running = False
                    self._idle.set()
                    return
            item = job[0]
            result: Any = None
            error: BaseException | None = None
            try:
                result = self._work(item)
            except BaseException as exc:  # el error se informa a la UI, no mata el hilo
                error = exc
            with self._lock:
                stale = self._pending is not None
            if not stale:
                self._on_done(item, result, error)

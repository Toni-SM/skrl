from __future__ import annotations

import warp as wp

from skrl import config


class ScopedCapture:
    def __init__(
        self,
        *,
        device: str | wp.context.Device | None = None,
        stream: wp.Stream | None = None,
        force_module_load: bool | None = None,
        external: bool = False,
        enabled: bool = True,
    ) -> None:
        """Context manager for capturing CUDA graphs.

        Adapted from Warp implementation of `warp.ScopedCapture <https://nvidia.github.io/warp/modules/runtime.html#warp.ScopedCapture>`_
        to support enabling/disabling the capture.

        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param stream: CUDA stream to capture on. If not specified, the default stream will be used.
        :param force_module_load: Whether to force loading of all kernels before capture.
        :param external: Whether the capture was already started externally.
        :param enabled: Whether to enable the capture. If disabled, the capture will be skipped and the graph will be None.
        """
        self._graph = None
        self._enabled = enabled
        if enabled:
            self._device = config.warp.parse_device(device)
            self._stream = stream
            self._force_module_load = force_module_load
            self._external = external
            self._active = False

    @property
    def graph(self) -> wp.context.Graph | None:
        """Captured graph"""
        return self._graph

    def __enter__(self) -> "ScopedCapture":
        """Begin capture of a CUDA graph"""
        if self._enabled:
            self._graph = None
            try:
                wp.capture_begin(
                    device=self._device,
                    stream=self._stream,
                    force_module_load=self._force_module_load,
                    external=self._external,
                )
                self._active = True
            except:
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """End capture of a CUDA graph"""
        if self._enabled:
            try:
                self._graph = wp.capture_end(device=self._device, stream=self._stream)
            finally:
                self._active = False

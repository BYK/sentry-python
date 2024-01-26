from __future__ import annotations

import socket
from sentry_sdk import Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration

if TYPE_CHECKING:
    from socket import AddressFamily, SocketKind
    from typing import Tuple, Optional, Union, List

__all__ = ["SocketIntegration"]


class SocketIntegration(Integration):
    identifier = "socket"

    @staticmethod
    def setup_once() -> None:
        """
        patches two of the most used functions of socket: create_connection and getaddrinfo(dns resolver)
        """
        _patch_create_connection()
        _patch_getaddrinfo()


def _get_span_description(
    host: Union[bytes, str, None], port: Union[str, int, None]
) -> str:
    try:
        host = host.decode()  # type: ignore
    except (UnicodeDecodeError, AttributeError):
        pass

    description = "%s:%s" % (host, port)  # type: ignore

    return description


def _patch_create_connection() -> None:
    real_create_connection = socket.create_connection

    def create_connection(
        address: Tuple[Optional[str], int],
        timeout: Optional[float] = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore
        source_address: Optional[Tuple[Union[bytearray, bytes, str], int]] = None,
    ) -> socket.socket:
        hub = Hub.current
        if hub.get_integration(SocketIntegration) is None:
            return real_create_connection(
                address=address, timeout=timeout, source_address=source_address
            )

        with hub.start_span(
            op=OP.SOCKET_CONNECTION,
            description=_get_span_description(address[0], address[1]),
        ) as span:
            span.set_data("address", address)
            span.set_data("timeout", timeout)
            span.set_data("source_address", source_address)

            return real_create_connection(
                address=address, timeout=timeout, source_address=source_address
            )

    socket.create_connection = create_connection  # type: ignore


def _patch_getaddrinfo() -> None:
    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(
        host: Union[bytes, str, None],
        port: Union[str, int, None],
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> List[
        Tuple[
            AddressFamily,
            SocketKind,
            int,
            str,
            Union[Tuple[str, int], Tuple[str, int, int, int]],
        ]
    ]:
        hub = Hub.current
        if hub.get_integration(SocketIntegration) is None:
            return real_getaddrinfo(host, port, family, type, proto, flags)

        with hub.start_span(
            op=OP.SOCKET_DNS, description=_get_span_description(host, port)
        ) as span:
            span.set_data("host", host)
            span.set_data("port", port)

            return real_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo  # type: ignore

from datetime import datetime

from opentelemetry import trace as otel_trace, context
from opentelemetry.trace.status import StatusCode

import sentry_sdk
from sentry_sdk.consts import SPANSTATUS, SPANDATA
from sentry_sdk.profiler.continuous_profiler import get_profiler_id
from sentry_sdk.utils import (
    get_current_thread_meta,
    logger,
    nanosecond_time,
)
from sentry_sdk._types import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any
    from typing import Iterator
    from typing import Optional
    from typing import overload
    from typing import ParamSpec
    from typing import Tuple
    from typing import Union
    from typing import TypeVar

    from typing_extensions import TypedDict

    P = ParamSpec("P")
    R = TypeVar("R")

    import sentry_sdk.profiler
    from sentry_sdk.scope import Scope
    from sentry_sdk._types import (
        MeasurementUnit,
    )

    class SpanKwargs(TypedDict, total=False):
        trace_id: str
        """
        The trace ID of the root span. If this new span is to be the root span,
        omit this parameter, and a new trace ID will be generated.
        """

        span_id: str
        """The span ID of this span. If omitted, a new span ID will be generated."""

        parent_span_id: str
        """The span ID of the parent span, if applicable."""

        same_process_as_parent: bool
        """Whether this span is in the same process as the parent span."""

        sampled: bool
        """
        Whether the span should be sampled. Overrides the default sampling decision
        for this span when provided.
        """

        op: str
        """
        The span's operation. A list of recommended values is available here:
        https://develop.sentry.dev/sdk/performance/span-operations/
        """

        description: str
        """A description of what operation is being performed within the span."""

        hub: Optional["sentry_sdk.Hub"]
        """The hub to use for this span. This argument is DEPRECATED. Please use the `scope` parameter, instead."""

        status: str
        """The span's status. Possible values are listed at https://develop.sentry.dev/sdk/event-payloads/span/"""

        containing_transaction: Optional["Transaction"]
        """The transaction that this span belongs to."""

        start_timestamp: Optional[Union[datetime, float]]
        """
        The timestamp when the span started. If omitted, the current time
        will be used.
        """

        scope: "sentry_sdk.Scope"
        """The scope to use for this span. If not provided, we use the current scope."""

        origin: str
        """
        The origin of the span.
        See https://develop.sentry.dev/sdk/performance/trace-origin/
        Default "manual".
        """

    class TransactionKwargs(SpanKwargs, total=False):
        name: str
        """Identifier of the transaction. Will show up in the Sentry UI."""

        source: str
        """
        A string describing the source of the transaction name. This will be used to determine the transaction's type.
        See https://develop.sentry.dev/sdk/event-payloads/transaction/#transaction-annotations for more information.
        Default "custom".
        """

        parent_sampled: bool
        """Whether the parent transaction was sampled. If True this transaction will be kept, if False it will be discarded."""

        baggage: "Baggage"
        """The W3C baggage header value. (see https://www.w3.org/TR/baggage/)"""

    ProfileContext = TypedDict(
        "ProfileContext",
        {
            "profiler_id": str,
        },
    )

BAGGAGE_HEADER_NAME = "baggage"
SENTRY_TRACE_HEADER_NAME = "sentry-trace"

# Transaction source
# see https://develop.sentry.dev/sdk/event-payloads/transaction/#transaction-annotations
TRANSACTION_SOURCE_CUSTOM = "custom"
TRANSACTION_SOURCE_URL = "url"
TRANSACTION_SOURCE_ROUTE = "route"
TRANSACTION_SOURCE_VIEW = "view"
TRANSACTION_SOURCE_COMPONENT = "component"
TRANSACTION_SOURCE_TASK = "task"

# These are typically high cardinality and the server hates them
LOW_QUALITY_TRANSACTION_SOURCES = [
    TRANSACTION_SOURCE_URL,
]

SOURCE_FOR_STYLE = {
    "endpoint": TRANSACTION_SOURCE_COMPONENT,
    "function_name": TRANSACTION_SOURCE_COMPONENT,
    "handler_name": TRANSACTION_SOURCE_COMPONENT,
    "method_and_path_pattern": TRANSACTION_SOURCE_ROUTE,
    "path": TRANSACTION_SOURCE_URL,
    "route_name": TRANSACTION_SOURCE_COMPONENT,
    "route_pattern": TRANSACTION_SOURCE_ROUTE,
    "uri_template": TRANSACTION_SOURCE_ROUTE,
    "url": TRANSACTION_SOURCE_ROUTE,
}

tracer = otel_trace.get_tracer(__name__)


def get_span_status_from_http_code(http_status_code):
    # type: (int) -> str
    """
    Returns the Sentry status corresponding to the given HTTP status code.

    See: https://develop.sentry.dev/sdk/event-payloads/contexts/#trace-context
    """
    if http_status_code < 400:
        return SPANSTATUS.OK

    elif 400 <= http_status_code < 500:
        if http_status_code == 403:
            return SPANSTATUS.PERMISSION_DENIED
        elif http_status_code == 404:
            return SPANSTATUS.NOT_FOUND
        elif http_status_code == 429:
            return SPANSTATUS.RESOURCE_EXHAUSTED
        elif http_status_code == 413:
            return SPANSTATUS.FAILED_PRECONDITION
        elif http_status_code == 401:
            return SPANSTATUS.UNAUTHENTICATED
        elif http_status_code == 409:
            return SPANSTATUS.ALREADY_EXISTS
        else:
            return SPANSTATUS.INVALID_ARGUMENT

    elif 500 <= http_status_code < 600:
        if http_status_code == 504:
            return SPANSTATUS.DEADLINE_EXCEEDED
        elif http_status_code == 501:
            return SPANSTATUS.UNIMPLEMENTED
        elif http_status_code == 503:
            return SPANSTATUS.UNAVAILABLE
        else:
            return SPANSTATUS.INTERNAL_ERROR

    return SPANSTATUS.UNKNOWN_ERROR


class Span:
    """
    A span holds timing information of a block of code.

    Spans can have multiple child spans thus forming a span tree.

    As of 3.0, this class is an OTel span wrapper providing compatibility
    with the old span interface. The wrapper itself should have as little state
    as possible. Everything persistent should be stored on the underlying OTel
    span.

    :param trace_id: The trace ID of the root span. If this new span is to be the root span,
        omit this parameter, and a new trace ID will be generated.
    :param span_id: The span ID of this span. If omitted, a new span ID will be generated.
    :param parent_span_id: The span ID of the parent span, if applicable.
    :param same_process_as_parent: Whether this span is in the same process as the parent span.
    :param sampled: Whether the span should be sampled. Overrides the default sampling decision
        for this span when provided.
    :param op: The span's operation. A list of recommended values is available here:
        https://develop.sentry.dev/sdk/performance/span-operations/
    :param description: A description of what operation is being performed within the span.
    :param status: The span's status. Possible values are listed at
        https://develop.sentry.dev/sdk/event-payloads/span/
    :param containing_transaction: The transaction that this span belongs to.
    :param start_timestamp: The timestamp when the span started. If omitted, the current time
        will be used.
    :param scope: The scope to use for this span. If not provided, we use the current scope.
    """

    def __init__(
        self,
        *,
        op=None,  # type: Optional[str]
        description=None,  # type: Optional[str]
        status=None,  # type: Optional[str]
        scope=None,  # type: Optional[Scope]
        start_timestamp=None,  # type: Optional[Union[datetime, float]]
        origin="manual",  # type: str
        **_,  # type: dict[str, object]
        # XXX old args:
        #         trace_id=None,  # type: Optional[str]
        #         span_id=None,  # type: Optional[str]
        #         parent_span_id=None,  # type: Optional[str]
        #         same_process_as_parent=True,  # type: bool
        #         sampled=None,  # type: Optional[bool]
        #         op=None,  # type: Optional[str]
        #         description=None,  # type: Optional[str]
        #         hub=None,  # type: Optional[sentry_sdk.Hub]  # deprecated
        #         status=None,  # type: Optional[str]
        #         containing_transaction=None,  # type: Optional[Transaction]
        #         start_timestamp=None,  # type: Optional[Union[datetime, float]]
        #         scope=None,  # type: Optional[sentry_sdk.Scope]
        #         origin="manual",  # type: str
    ):
        # type: (...) -> None
        """
        For backwards compatibility with the old Span interface, this class
        accepts arbitrary keyword arguments, in addition to the ones explicitly
        listed in the signature. These additional arguments are ignored.
        """
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute
        from sentry_sdk.integrations.opentelemetry.utils import (
            convert_to_otel_timestamp,
        )

        if start_timestamp is not None:
            # OTel timestamps have nanosecond precision
            start_timestamp = convert_to_otel_timestamp(start_timestamp)

        self._otel_span = tracer.start_span(
            description or op or "", start_time=start_timestamp
        )  # XXX

        # XXX deal with _otel_span being a NonRecordingSpan

        self._otel_span.set_attribute(SentrySpanAttribute.ORIGIN, origin)

        if op is not None:
            self.op = op

        self.description = description

        if status is not None:
            self.set_status(status)

        self.scope = scope

        try:
            # profiling depends on this value and requires that
            # it is measured in nanoseconds
            self._start_timestamp_monotonic_ns = nanosecond_time()
        except AttributeError:
            pass

        self._local_aggregator = None  # type: Optional[LocalAggregator]

        thread_id, thread_name = get_current_thread_meta()
        self.set_thread(thread_id, thread_name)
        self.set_profiler_id(get_profiler_id())

    def __repr__(self):
        # type: () -> str
        return (
            "<%s(op=%r, description:%r, trace_id=%r, span_id=%r, parent_span_id=%r, sampled=%r, origin=%r)>"
            % (
                self.__class__.__name__,
                self.op,
                self.description,
                self.trace_id,
                self.span_id,
                self.parent_span_id,
                self.sampled,
                self.origin,
            )
        )

    def __enter__(self):
        # type: () -> Span
        # XXX use_span? https://github.com/open-telemetry/opentelemetry-python/blob/3836da8543ce9751051e38a110c0468724042e62/opentelemetry-api/src/opentelemetry/trace/__init__.py#L547
        #
        # create a Context object with parent set as current span
        ctx = otel_trace.set_span_in_context(self._otel_span)
        # set as the implicit current context
        self._ctx_token = context.attach(ctx)
        scope = self.scope or sentry_sdk.get_current_scope()
        scope.span = self
        return self

    def __exit__(self, ty, value, tb):
        # type: (Optional[Any], Optional[Any], Optional[Any]) -> None
        self._otel_span.end()
        # XXX set status to error if unset and an exception occurred?
        context.detach(self._ctx_token)

    @property
    def name(self):
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def source(self):
        pass

    @source.setter
    def source(self, value):
        pass

    @property
    def description(self):
        # type: () -> Optional[str]
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        return self._otel_span.attributes.get(SentrySpanAttribute.DESCRIPTION)

    @description.setter
    def description(self, value):
        # type: (Optional[str]) -> None
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        if value is not None:
            self._otel_span.set_attribute(SentrySpanAttribute.DESCRIPTION, value)

    @property
    def origin(self):
        # type: () -> Optional[str]
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        return self._otel_span.attributes.get(SentrySpanAttribute.ORIGIN)

    @origin.setter
    def origin(self, value):
        # type: (Optional[str]) -> None
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        if value is not None:
            self._otel_span.set_attribute(SentrySpanAttribute.ORIGIN, value)

    @property
    def root_span(self):
        if isinstance(self._otel_span, otel_trace.NonRecordingSpan):
            return None

        parent = None
        while True:
            # XXX
            if self._otel_span.parent:
                parent = self._otel_span.parent
            else:
                break

        return parent

    @property
    def is_segment(self):
        if isinstance(self._otel_span, otel_trace.NonRecordingSpan):
            return False

        return self._otel_span.parent is None

    @property
    def containing_transaction(self):
        # type: () -> Optional[Span]
        """
        Get the transaction this span is a child of.

        .. deprecated:: 3.0.0
            This will be removed in the future.
        """

        logger.warning("Deprecated: This will be removed in the future.")
        return self.root_span

    @containing_transaction.setter
    def containing_transaction(self, value):
        # type: (Span) -> None
        """
        Set this span's transaction.

        .. deprecated:: 3.0.0
            Use :func:`root_span` instead.
        """
        pass

    @property
    def parent_span_id(self):
        # type: () -> Optional[str]
        return self._otel_span.parent if hasattr(self._otel_span, "parent") else None

    @property
    def trace_id(self):
        # type: () -> Optional[str]
        return self._otel_span.get_span_context().trace_id

    @property
    def span_id(self):
        # type: () -> Optional[str]
        return self._otel_span.get_span_context().span_id

    @property
    def sampled(self):
        # type: () -> Optional[bool]
        return self._otel_span.get_span_context().trace_flags.sampled

    @sampled.setter
    def sampled(self, value):
        # type: () -> Optional[bool]
        pass

    @property
    def op(self):
        # type: () -> Optional[str]
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        self._otel_span.attributes.get(SentrySpanAttribute.OP)

    @op.setter
    def op(self, value):
        # type: (str) -> None
        from sentry_sdk.integrations.opentelemetry.consts import SentrySpanAttribute

        self._otel_span.set_attribute(SentrySpanAttribute.OP, value)

    def start_child(self, **kwargs):
        # type: (str, **Any) -> Span
        kwargs.setdefault("sampled", self.sampled)
        child_span = Span(**kwargs)
        return child_span

    @classmethod
    def continue_from_environ(
        cls,
        environ,  # type: Mapping[str, str]
        **kwargs,  # type: Any
    ):
        # type: (...) -> Span
        # XXX actually propagate
        span = Span(**kwargs)
        return span

    @classmethod
    def continue_from_headers(
        cls,
        headers,  # type: Mapping[str, str]
        **kwargs,  # type: Any
    ):
        # type: (...) -> Span
        # XXX actually propagate
        span = Span(**kwargs)
        return span

    def iter_headers(self):
        # type: () -> Iterator[Tuple[str, str]]
        pass

    @classmethod
    def from_traceparent(
        cls,
        traceparent,  # type: Optional[str]
        **kwargs,  # type: Any
    ):
        # type: (...) -> Optional[Span]
        # XXX actually propagate
        span = Span(**kwargs)
        return span

    def to_traceparent(self):
        # type: () -> str
        if self.sampled is True:
            sampled = "1"
        elif self.sampled is False:
            sampled = "0"
        else:
            sampled = None

        traceparent = "%s-%s" % (self.trace_id, self.span_id)
        if sampled is not None:
            traceparent += "-%s" % (sampled,)

        return traceparent

    def to_baggage(self):
        # type: () -> Optional[Baggage]
        pass

    def set_tag(self, key, value):
        # type: (str, Any) -> None
        pass

    def set_data(self, key, value):
        # type: (str, Any) -> None
        self._otel_span.set_attribute(key, value)

    def set_status(self, status):
        # type: (str) -> None
        if status == SPANSTATUS.OK:
            otel_status = StatusCode.OK
            otel_description = None
        else:
            otel_status = StatusCode.ERROR
            otel_description = status.value

        self._otel_span.set_status(otel_status, otel_description)

    def set_measurement(self, name, value, unit=""):
        # type: (str, float, MeasurementUnit) -> None
        # XXX own namespace, e.g. sentry.measurement.xxx?
        self._otel_span.set_attribute(name, (value, unit))

    def set_thread(self, thread_id, thread_name):
        # type: (Optional[int], Optional[str]) -> None
        if thread_id is not None:
            self.set_data(SPANDATA.THREAD_ID, str(thread_id))

            if thread_name is not None:
                self.set_data(SPANDATA.THREAD_NAME, thread_name)

    def set_profiler_id(self, profiler_id):
        # type: (Optional[str]) -> None
        if profiler_id is not None:
            self.set_data(SPANDATA.PROFILER_ID, profiler_id)

    def set_http_status(self, http_status):
        # type: (int) -> None
        self.set_tag(
            "http.status_code", str(http_status)
        )  # we keep this for backwards compatibility
        # XXX do we still need this? ^
        self.set_data(SPANDATA.HTTP_STATUS_CODE, http_status)
        self.set_status(get_span_status_from_http_code(http_status))

    def is_success(self):
        # type: () -> bool
        return self._otel_span.status.code == StatusCode.OK

    def finish(self, scope=None, end_timestamp=None):
        # type: (Optional[sentry_sdk.Scope], Optional[Union[float, datetime]]) -> Optional[str]
        # XXX check if already finished
        from sentry_sdk.integrations.opentelemetry.utils import (
            convert_to_otel_timestamp,
        )

        if end_timestamp is not None:
            end_timestamp = convert_to_otel_timestamp(end_timestamp)
        self._otel_span.end(end_time=end_timestamp)
        scope = scope or sentry_sdk.get_current_scope()
        maybe_create_breadcrumbs_from_span(scope, self)

    def to_json(self):
        # type: () -> dict[str, Any]
        pass

    def get_trace_context(self):
        # type: () -> Any
        pass

    def get_profile_context(self):
        # type: () -> Optional[ProfileContext]
        pass

    def _get_local_aggregator(self):
        # type: (...) -> LocalAggregator
        rv = self._local_aggregator
        if rv is None:
            rv = self._local_aggregator = LocalAggregator()
        return rv

    # transaction/root span methods

    def set_context(self, key, value):
        # type: (str, Any) -> None
        pass

    def get_baggage(self):
        # type: () -> Baggage
        pass


Transaction = Span


class NoOpSpan:
    # XXX
    pass


if TYPE_CHECKING:

    @overload
    def trace(func=None):
        # type: (None) -> Callable[[Callable[P, R]], Callable[P, R]]
        pass

    @overload
    def trace(func):
        # type: (Callable[P, R]) -> Callable[P, R]
        pass


def trace(func=None):
    # type: (Optional[Callable[P, R]]) -> Union[Callable[P, R], Callable[[Callable[P, R]], Callable[P, R]]]
    """
    Decorator to start a child span under the existing current transaction.
    If there is no current transaction, then nothing will be traced.

    .. code-block::
        :caption: Usage

        import sentry_sdk

        @sentry_sdk.trace
        def my_function():
            ...

        @sentry_sdk.trace
        async def my_async_function():
            ...
    """
    from sentry_sdk.tracing_utils import start_child_span_decorator

    # This patterns allows usage of both @sentry_traced and @sentry_traced(...)
    # See https://stackoverflow.com/questions/52126071/decorator-with-arguments-avoid-parenthesis-when-no-arguments/52126278
    if func:
        return start_child_span_decorator(func)
    else:
        return start_child_span_decorator


# Circular imports

from sentry_sdk.tracing_utils import (
    Baggage,
    maybe_create_breadcrumbs_from_span,
)
from sentry_sdk.metrics import LocalAggregator

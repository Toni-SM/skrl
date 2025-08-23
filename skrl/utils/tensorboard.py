import time
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class SummaryWriter:
    def __init__(self, log_dir: str, *, queue_size: int = 10, flush_seconds: int = 120):
        """TensorBoard summary writer.

        :param log_dir: Directory to save the event file.
        :param queue_size: Maximum number of events to keep in the queue before forcing a flush to persistent storage.
        :param flush_seconds: Number of seconds between flushing pending events to persistent storage.
        """
        self._event_file_writer = EventFileWriter(log_dir, max_queue_size=queue_size, flush_secs=flush_seconds)

    def add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        """Add a scalar value to the summary.

        :param tag: Name of the scalar record.
        :param value: Value of the scalar.
        :param timestep: Global step value to record.
        """
        event = Event(summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]))
        event.step = timestep
        event.wall_time = time.time()
        self._event_file_writer.add_event(event)

    def flush(self) -> None:
        """Flush pending events to persistent storage."""
        self._event_file_writer.flush()

    def close(self) -> None:
        """Close the summary writer."""
        self._event_file_writer.close()

#!/usr/bin/env python3

from __future__ import annotations

import errno
import os
import sys
import time
from math import inf
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Queue as MPQueue
from pathlib import Path
from queue import Empty

import click
import serial
from asserttool import ic
from asserttool import icp
from clicktool import click_add_options
from clicktool import click_global_options
from clicktool import tvicgvd
from eprint import eprint
from globalverbose import gvd
from serial.tools import list_ports
from timestamptool import get_int_timestamp
from timestamptool import get_timestamp

DATA_DIR = Path(Path(os.path.expanduser("~")) / Path(".serialtool"))
DATA_DIR.mkdir(exist_ok=True)


class SerialNoResponseError(ValueError):
    pass


def construct_serial_command(
    command: bytes,
    argument: None | bytes = None,
):
    if argument:
        command = command + argument

    command = b"\x10\x02" + command + b"\x10\x03"
    return command


def construct_serial_command_ack(
    command: bytes,
):
    if gvd:
        ic(command)
    return b"\x06" + command


def generate_serial_port_help():
    help_text = "Available serial ports: "
    ports = list_ports.comports()
    _ports = [str(port) for port in ports]
    help_text = repr(tuple(_ports))
    help_text = help_text.replace("\n\n", "\n")
    return help_text


def wait_for_serial_queue(*, serial_queue, timeout: float):
    ic("waiting for self.serial_queue.qsize() > 0")
    start = time.time()
    while True:
        if serial_queue.qsize() > 0:
            ic(serial_queue.qsize())
            break
        if (time.time() - start) > timeout:
            raise TimeoutError("nothing arrived in the serial_queue in time")


def pick_serial_port() -> Path:
    ftdi_devices = ["US232R", "USB-Serial Controller"]
    ports = list_ports.comports()
    for port in ports:
        port_str = str(port)
        ic(port_str)
        for device in ftdi_devices:
            if port_str.endswith(device):
                return Path(port_str.split(" ")[0])

    for port in ports:
        port_str = str(port)
        if not port_str.endswith("n/a"):
            return Path(port_str.split(" ")[0])

    print("")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print(
        "------------------------!!!!VERIFY USB SERIAL IS PLUGGED IN!!!!---------------------"
    )
    print(
        f"Didnt find a port with a connected FTDI device from the list: {ftdi_devices}"
    )
    print("picking the first port instead.", file=sys.stderr)
    print("")
    try:
        port = str(ports[0]).split(" ")[0]
    except IndexError:
        ic("No serial ports were found. Exiting.")
        sys.exit(1)
    return Path(port)


def print_serial_oracle(
    *,
    serial_oracle: SerialOracle,
    timestamp: bool,
    read_tx_from_fifo: bool,
    show_bytes: bool = False,
):
    last_queue_size = None
    queue_size = 0

    while True:
        if gvd:
            queue_size = serial_oracle.rx_queue.qsize()
            if queue_size != last_queue_size:
                ic(queue_size)
                last_queue_size = queue_size
        try:
            data = serial_oracle.rx_queue.get(False)
            data = data[0]
            if show_bytes:
                ic(data)
            if timestamp:
                _timestamp = get_timestamp()
                data = _timestamp.encode("utf8") + b" " + data
                if gvd:
                    ic(data)
            byte_count_written_to_stdout = sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if gvd:
                ic(byte_count_written_to_stdout)
        except Empty:
            pass


class SerialMinimal:
    __slots__ = (
        "log_serial_data",
        "serial_port",
        "baud_rate",
        "default_timeout",
        "hardware_buffer_size",
        "data_dir",
        "serial_data_dir",
        "ser",
    )

    def __init__(
        self,
        log_serial_data: bool,
        serial_port: str,
        baud_rate: int,
        default_timeout: float = 1.0,
        hardware_buffer_size: int = 4096,
        data_dir: Path = DATA_DIR,
    ):
        self.log_serial_data = log_serial_data
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.default_timeout = default_timeout
        self.hardware_buffer_size = hardware_buffer_size
        self.data_dir = data_dir

        ic(self.serial_port)
        self.serial_data_dir = self.data_dir / "serial_logs"
        self.serial_data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_int_timestamp()
        serial_data_file = (
            self.serial_data_dir / f"{timestamp}_{Path(self.serial_port).name}"
        )

        serial_url_list = ["spy://", self.serial_port, "?file="]
        if self.log_serial_data:
            serial_url_list.append(serial_data_file.as_posix())
        else:
            serial_url_list.append("/dev/null" if sys.platform == "linux" else "NUL:")

        serial_url = "".join(serial_url_list)
        eprint(f"{serial_url=}")
        icp(self.serial_data_dir) if self.log_serial_data else None

        self.ser = serial.serial_for_url(serial_url)
        self.ser.baudrate = self.baud_rate
        self.ser.timeout = self.default_timeout
        self.ser.ctsrts = False
        self.ser.dsrdtr = False
        self.ser.xonxoff = False

        ic(
            self.ser.port,
            self.ser.baudrate,
            self.ser.bytesize,
            self.ser.parity,
            self.ser.stopbits,
            self.ser.timeout,
            self.ser.xonxoff,
            self.ser.rtscts,
            self.ser.dsrdtr,
            self.ser.write_timeout,
            self.ser.inter_byte_timeout,
            self.ser.exclusive,
        )

        discard = self.ser.readall()
        if gvd:
            ic(discard)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()


class SerialQueue:
    __slots__ = (
        "rx_queue",
        "tx_queue",
        "serial_data_dir",
        "log_serial_data",
        "ready_signal",
        "serial_port",
        "terse",
        "baud_rate",
        "default_timeout",
        "hardware_buffer_size",
        "ser",
    )

    def __init__(
        self,
        rx_queue: MPQueue,
        tx_queue: MPQueue,
        serial_data_dir: Path,
        log_serial_data: bool,
        ready_signal: str,
        serial_port: Path,
        terse: bool,
        baud_rate: int = 460800,
        default_timeout: float = 1.0,
        hardware_buffer_size: int = 4096,
    ):
        self.rx_queue = rx_queue
        self.tx_queue = tx_queue
        self.serial_data_dir = serial_data_dir
        self.log_serial_data = log_serial_data
        self.ready_signal = ready_signal
        self.serial_port = serial_port
        self.terse = terse
        self.baud_rate = baud_rate
        self.default_timeout = default_timeout
        self.hardware_buffer_size = hardware_buffer_size

    def listen_serial(self):
        if not self.serial_port:
            ic("No serial port specified, picking:")
            self.serial_port = pick_serial_port()

        ic(self.serial_port)
        serial_data_dir = self.serial_data_dir / "serial_logs"
        serial_data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_int_timestamp()
        serial_data_file = serial_data_dir / f"{timestamp}_{self.serial_port.name}"

        serial_url_list = ["spy://", self.serial_port.as_posix(), "?file="]
        if self.log_serial_data:
            serial_url_list.append(serial_data_file.as_posix())
        else:
            serial_url_list.append("/dev/null" if sys.platform == "linux" else "NUL:")

        serial_url = "".join(serial_url_list)
        eprint(f"{serial_url=}")
        icp(self.serial_data_dir) if self.log_serial_data else None

        self.ser = serial.serial_for_url(serial_url)
        self.ser.baudrate = self.baud_rate
        self.ser.timeout = self.default_timeout
        self.ser.ctsrts = False
        self.ser.dsrdtr = False
        self.ser.xonxoff = False

        discard = self.ser.readall()
        if gvd:
            ic(discard)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.rx_queue.put([self.ready_signal])
        while True:
            read_bytes = self.ser.read(self.ser.inWaiting() or 1)
            if len(read_bytes) == self.hardware_buffer_size:
                raise ValueError(
                    f"serial hardware overflow: {self.hardware_buffer_size=}, {self.rx_queue.qsize()=}"
                )

            if read_bytes:
                self.rx_queue.put([read_bytes])

            if self.ser.inWaiting() == 0:
                while self.tx_queue.qsize() > 0:
                    try:
                        item = self.tx_queue.get(False)
                        if item == ["EXIT"]:
                            icp("got [EXIT]")
                            sys.exit(0)
                        data = item[0]
                        written = self.ser.write(data)
                        assert written == len(data)
                        self.ser.flush()
                    except Empty:
                        pass


def launch_serial_queue_process(
    *,
    rx_queue: Queue,
    tx_queue: Queue,
    serial_port: Path,
    serial_data_dir: Path,
    baud_rate: int,
    log_serial_data: bool,
    terse: bool,
):
    ready_signal = str(time.time())
    serial_queue = SerialQueue(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_data_dir=serial_data_dir,
        log_serial_data=log_serial_data,
        ready_signal=ready_signal,
        serial_port=serial_port,
        baud_rate=baud_rate,
        terse=terse,
    )
    serial_process = Process(
        target=serial_queue.listen_serial,
        daemon=True,
    )
    serial_process.start()

    wait_for_serial_queue(serial_queue=rx_queue, timeout=15.0)
    ready_string = rx_queue.get(timeout=1.0)[0]
    assert ready_string == ready_signal
    return serial_process


def print_serial_output(
    *,
    serial_port: Path,
    serial_data_dir: Path,
    log_serial_data: bool,
    timestamp: bool,
    show_bytes: bool,
    baud_rate: int,
    read_tx_from_fifo: bool,
    terse: bool,
):
    if not serial_port:
        serial_port = pick_serial_port()

    icp(serial_port)
    rx_queue = Queue()
    tx_queue = Queue()

    _ = launch_serial_queue_process(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_port=serial_port,
        serial_data_dir=serial_data_dir,
        baud_rate=baud_rate,
        log_serial_data=log_serial_data,
        terse=terse,
    )

    if read_tx_from_fifo:
        tx_fifo_path = DATA_DIR / "tx_fifo"
        if not tx_fifo_path.exists():
            os.mkfifo(tx_fifo_path.as_posix())
        tx_fifo = os.open(tx_fifo_path.as_posix(), os.O_RDONLY | os.O_NONBLOCK)

    while True:
        if read_tx_from_fifo:
            # non-blocking
            # https://stackoverflow.com/questions/47116027/reading-input-from-a-fifo-without-blocking
            try:
                data = os.read(tx_fifo, 1024)
            except OSError as err:
                if err.errno == errno.EAGAIN or err.errno == errno.EWOULDBLOCK:
                    data = None
                else:
                    raise

            if data:
                tx_queue.put([data])
                eprint(f"sent: {data}")

        try:
            line_bytes = rx_queue.get(timeout=0.1)
        except Empty:
            continue

        line_bytes = line_bytes[0]

        if timestamp:
            timestamp = get_timestamp()
            sys.stdout.buffer.write(timestamp.encode("utf-8"))
            sys.stdout.buffer.write(b": ")

        if show_bytes:
            sys.stdout.buffer.write(str(line_bytes).encode("utf-8"))
        else:
            sys.stdout.buffer.write(line_bytes)

        sys.stdout.buffer.flush()


class SerialOracle:
    __slots__ = (
        "rx_queue",
        "tx_queue",
        "serial_port",
        "serial_process",
        "data_dir",
        "terse",
        "baud_rate",
        "log_serial_data",
        "display_communication",
        "ipython_on_communication_error",
    )

    def __init__(
        self,
        serial_port: Path,
        serial_data_dir: Path,
        terse: bool,
        baud_rate: int = 460800,
        log_serial_data: bool = False,
        display_communication: bool = False,
        ipython_on_communication_error: bool = False,
    ):
        self.rx_queue = Queue()
        self.tx_queue = Queue()
        self.serial_port = serial_port
        self.data_dir = serial_data_dir
        self.terse = terse
        self.baud_rate = baud_rate
        self.log_serial_data = log_serial_data
        self.display_communication = display_communication
        self.ipython_on_communication_error = ipython_on_communication_error

        self.serial_process = launch_serial_queue_process(
            rx_queue=self.rx_queue,
            tx_queue=self.tx_queue,
            serial_port=self.serial_port,
            serial_data_dir=self.data_dir,
            baud_rate=self.baud_rate,
            log_serial_data=self.log_serial_data,
            terse=self.terse,
        )

    def _write(self, data: bytes):
        if not self.terse:
            eprint(
                f"serialtool: _write() writing {len(data)=} bytes to self.tx_queue: {repr(data)=}"
            )
        else:
            if gvd:
                eprint(
                    f"serialtool: _write() writing {len(data)=} bytes to self.tx_queue: {repr(data)=}"
                )
        self.tx_queue.put([data])

    def _read(
        self,
        *,
        count: int | float,
        progress: bool = False,
    ) -> bytes:
        """
        Read count number of bytes from the serial RX queue.
        count = inf:
            read until manually stopped
        """
        read_bytes = b""
        time_limit = 1.0

        # return read_bytes
        if not self.terse:
            eprint(f"serialtool: _read() {count=}")

        if count == 0:
            read_bytes = b""
        elif count == inf:
            raise ValueError(count)
            # eprint("_read() count == inf ERROR (todo)")
        elif count > 0:
            start_time = time.time()
            while len(read_bytes) < count:
                try:
                    time_to_wait = min(
                        0.1, time_limit - (time.time() - start_time)
                    )  # time_limit == 1.0 rn
                    queue_data = self.rx_queue.get(timeout=time_to_wait)
                    read_bytes += queue_data[0]
                    start_time = time.time()  # reset the timeout on new data
                    if progress:
                        eprint(f"{len(read_bytes)}/{count}\r", end="")
                except Empty as e:
                    # ic(e)
                    raise e
                    # print("_read() got Empty", flush=True)
                    # pass
        else:
            raise ValueError(count)
        if not self.terse:
            if gvd:
                eprint(
                    f"serialtool: _read() got {len(read_bytes)} bytes: {repr(read_bytes)=}"
                )
            else:
                if len(read_bytes) < 1000:
                    eprint(
                        f"serialtool: _read() got {len(read_bytes)} bytes: {repr(read_bytes)=}"
                    )
                else:
                    eprint(
                        f"serialtool: _read() got {len(read_bytes)} bytes: (truncated){repr(read_bytes)[:100]=}"
                    )
        # ic(len(read_bytes), read_bytes)
        return read_bytes

    def send_serial_command_queued(
        self,
        command: bytes,
        *,
        byte_count_requested: None | int,
        expect_ack: bool,
        echo: bool = True,
        timeout: None | float = None,
        progress: bool = False,
    ):
        if not timeout:
            timeout = inf

        command = construct_serial_command(command=command)
        ic(command, byte_count_requested)

        if byte_count_requested == 0:
            expected_response_bytes = None
        else:
            ic(command)
            expected_response_bytes = b"\x10\x02" + command[2:4] + b"\x10\x03"
            if expect_ack:
                expected_response_bytes = (
                    expected_response_bytes
                    + b"\x06"
                    + construct_serial_command_ack(command=command[2:4])
                )

            byte_count_requested += len(expected_response_bytes)
            ic(expected_response_bytes, len(expected_response_bytes))

        start_time = time.time()
        self._write(data=command)

        rx_bytes = self.read_command_result(
            byte_count_requested=byte_count_requested,
            bytes_expected=expected_response_bytes,
            timeout=timeout,
            progress=progress,
        )

        result_bytes = self.extract_command_result(
            two_byte_command=command[2:4],
            result=rx_bytes,
            expect_ack=expect_ack,
            data_bytes_expected=byte_count_requested - len(expected_response_bytes),
        )

        if self.display_communication:
            if echo:
                if self.terse:
                    eprint(f"{repr(result_bytes)=}")
                else:
                    eprint(
                        f"serialtool: send_serial_command_queued() {len(result_bytes)=} {repr(result_bytes)=}"
                    )
        return result_bytes

    def send_serial_command_direct(
        self,
        command: bytes,
        *,
        byte_count_requested: None | int = None,
        expected_response: None | bytes = None,
        echo: bool = True,
        timeout: None | float = None,
        progress: bool = False,
    ):
        if not timeout:
            timeout = inf

        if byte_count_requested:
            ic(byte_count_requested)
            assert isinstance(byte_count_requested, int)

        if expected_response:
            ic(expected_response)
            assert isinstance(expected_response, bytes)

        start_time = time.time()
        self._write(data=command)
        try:
            rx_bytes = self.read_command_result(
                byte_count_requested=byte_count_requested,
                bytes_expected=expected_response,
                timeout=timeout,
                progress=progress,
            )
        except ValueError as e:
            ic(e)
            raise e

        if expected_response:
            if rx_bytes != expected_response:
                ic(rx_bytes)
                ic(expected_response)
                raise ValueError(rx_bytes)
        if echo:
            if self.terse:
                eprint(f"{repr(rx_bytes)=}")
            else:
                eprint(
                    f"serialtool: send_serial_command_direct() {len(rx_bytes)=} {repr(rx_bytes)=}"
                )
        return rx_bytes

    def read_command_result(
        self,
        *,
        byte_count_requested: None | int = None,
        bytes_expected: None | bytes = None,
        timeout: None | float = None,
        progress: bool = False,
    ):
        """
        Reads byte_count_requested number of bytes over serial and compares it to bytes_expected if given.
        bytes_expected = None:
            no bytes expected to be read back
        """
        if bytes_expected:
            assert isinstance(bytes_expected, bytes)

        if bytes_expected:
            assert byte_count_requested is None
            byte_count_requested = len(bytes_expected)

        if not timeout:
            timeout = inf

        if byte_count_requested == 0:
            assert bytes_expected is None
            bytes_expected = b""
            assert self.rx_queue.qsize() == 0
            return b""  # Fixed: was returning None implicitly

        if byte_count_requested == inf:
            assert False
            assert timeout > 0
            assert bytes_expected is None
            all_bytes = b""
            start_time = time.time()
            ic(start_time, timeout)
            while True:
                read_bytes = self._read(count=inf)
                if gvd and read_bytes:
                    ic(read_bytes)
                all_bytes += read_bytes
                if (time.time() - start_time) > timeout:
                    ic("TIMEOUT", timeout)
                    raise TimeoutError(timeout)
            return all_bytes

        result = b""
        start_time = time.time()
        while len(result) < byte_count_requested:
            bytes_needed = byte_count_requested - len(result)

            if bytes_needed > 0:
                try:
                    result += self._read(count=bytes_needed, progress=progress)
                except Empty as e:
                    pass

            if (time.time() - start_time) > timeout:
                raise TimeoutError(timeout)
        if progress:
            eprint(f"\ndone: {len(result)}/{byte_count_requested}\n")
        _duration = time.time() - start_time
        if _duration > 0.25:
            _bytes_per_second = int(len(result) / _duration)
            _bits_per_second = _bytes_per_second * 8
            eprint(f"{_duration=}, {_bytes_per_second=}, {_bits_per_second=}")

        if bytes_expected:
            if len(result) == 0:
                raise SerialNoResponseError()

            if result != bytes_expected:
                icp(byte_count_requested)
                icp(repr(bytes_expected), len(bytes_expected))
                icp(repr(result), len(result))
                if self.ipython_on_communication_error:
                    import IPython

                    IPython.embed()
                ic("About to raise ValueError on result:", result)
                raise ValueError(result)
        if len(result) < 10:
            ic(result[0:10])
        elif gvd:
            ic(result)
        return result

    def extract_command_result(
        self,
        two_byte_command: bytes,
        result: bytes,
        expect_ack: bool,
        data_bytes_expected: int,
    ):
        if len(result) > 100:
            if gvd:
                eprint(
                    f"serialtool: extract_command_result() {two_byte_command=} {result=} {expect_ack=} {data_bytes_expected=}"
                )
            else:
                if self.terse:
                    pass
                else:
                    eprint(
                        f"serialtool: extract_command_result() {two_byte_command=} (truncated){result[:100]=} {expect_ack=} {data_bytes_expected=}"
                    )
        assert isinstance(two_byte_command, bytes)
        if expect_ack:
            ending_bytes_expected = b"\x06" + two_byte_command
            if not self.terse:
                if len(result) < 100:
                    eprint(
                        f"serialtool: extract_command_result() {expect_ack=} {-len(ending_bytes_expected)=} {result[-len(ending_bytes_expected) :]=} {ending_bytes_expected=}"
                    )
                else:
                    if gvd:
                        eprint(
                            f"serialtool: extract_command_result() {expect_ack=} {-len(ending_bytes_expected)=} {result[-len(ending_bytes_expected) :]=} {ending_bytes_expected=}"
                        )
                    else:
                        eprint(
                            f"serialtool: extract_command_result() {expect_ack=} {-len(ending_bytes_expected)=} (truncated){result[-len(ending_bytes_expected) :100]=} {ending_bytes_expected=}"
                        )
            assert result[-len(ending_bytes_expected) :] == ending_bytes_expected
            result = result[:-3]

        if data_bytes_expected:
            assert len(result) - 4 == data_bytes_expected
            assert result[0:2] == b"\x10\x02"
            result = result[2:]
            assert result[-2:] == b"\x10\x03"
            result = result[:-2]
        return result


@click.command()
@click.argument(
    "serial_port",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path,
        allow_dash=False,
    ),
    nargs=1,
)
@click.option(
    "--data_dir",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        path_type=Path,
        allow_dash=False,
    ),
    default=DATA_DIR,
)
@click.option("--show-bytes", is_flag=True)
@click.option(
    "--baud-rate",
    type=int,
    default=460800,
)
@click.option("--timestamp", is_flag=True)
@click.option("--read-from-fifo", is_flag=True)
@click.option("--log-serial-data", is_flag=True)
@click.option("--terse", is_flag=True)
@click_add_options(click_global_options)
@click.pass_context
def cli(
    ctx,
    serial_port: Path,
    data_dir: Path,
    show_bytes: bool,
    log_serial_data: bool,
    baud_rate: int,
    read_from_fifo: bool,
    terse: bool,
    timestamp: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
):
    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    if not serial_port:
        serial_port = pick_serial_port()

    print_serial_output(
        serial_port=serial_port,
        serial_data_dir=data_dir,
        log_serial_data=log_serial_data,
        timestamp=timestamp,
        show_bytes=show_bytes,
        baud_rate=baud_rate,
        read_tx_from_fifo=read_from_fifo,
        terse=terse,
    )

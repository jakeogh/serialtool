#!/usr/bin/env python3
# -*- coding: utf8 -*-

# pylint: disable=useless-suppression             # [I0021]
# pylint: disable=missing-docstring               # [C0111] docstrings are always outdated and wrong
# pylint: disable=fixme                           # [W0511] todo is encouraged
# pylint: disable=line-too-long                   # [C0301]
# pylint: disable=too-many-instance-attributes    # [R0902]
# pylint: disable=too-many-lines                  # [C0302] too many lines in module
# pylint: disable=invalid-name                    # [C0103] single letter var names, name too descriptive
# pylint: disable=too-many-return-statements      # [R0911]
# pylint: disable=too-many-branches               # [R0912]
# pylint: disable=too-many-statements             # [R0915]
# pylint: disable=too-many-arguments              # [R0913]
# pylint: disable=too-many-nested-blocks          # [R1702]
# pylint: disable=too-many-locals                 # [R0914]
# pylint: disable=too-few-public-methods          # [R0903]
# pylint: disable=no-member                       # [E1101] no member for base
# pylint: disable=attribute-defined-outside-init  # [W0201]

from __future__ import annotations

import errno
import os
import sys
import time
from math import inf
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from queue import Empty

import attr
import click
import serial
from asserttool import ic
from asserttool import icp
from clicktool import click_add_options
from clicktool import click_global_options
from clicktool import tvicgvd
from cycloidal_client.ctia_construct_serial_command_and_response import \
    construct_serial_command
from cycloidal_client.exceptions import SerialNoResponseError
from eprint import eprint
from globalverbose import gvd
from serial.tools import list_ports
from timestamptool import get_int_timestamp
from timestamptool import get_timestamp

# from contextlib import ExitStack
# from shutil import get_terminal_size
# from multiprocessing import get_context

# import subprocess
# import atexit

gvd.disable()

DATA_DIR = Path(Path(os.path.expanduser("~")) / Path(".cycloidal_client"))
DATA_DIR.mkdir(exist_ok=True)


# def construct_serial_command(
#    command: bytes,
#    argument: None | bytes = None,
# ):
#    command_name = lookup_two_byte_command_name(two_bytes=command)
#    ic(command, command_name)
#    if argument:
#        command = command + argument
#
#    command = b"\x10\x02" + command + b"\x10\x03"
#    return command


def generate_serial_port_help():
    help_text = "Available serial ports: "
    ports = list_ports.comports()
    _ports = [str(port) for port in ports]
    help_text = repr(tuple(_ports))
    help_text.replace("\n\n", "\n")
    return help_text


def wait_for_serial_queue(
    *,
    serial_queue,
    timeout: float,
):
    ic("waiting for self.serial_queue.qsize() > 0")
    serial_queue_start_time = time.time()
    while True:
        if serial_queue.qsize() > 0:
            ic(serial_queue.qsize())
            break
        if (time.time() - serial_queue_start_time) > timeout:
            raise TimeoutError("nothing arrived in the serial_queue in time")


def pick_serial_port():
    ftdi_devices = ["US232R", "USB-Serial Controller"]
    ports = list_ports.comports()
    # ic(ports)
    for port in ports:
        port_str = str(port)
        ic(port_str)
        for device in ftdi_devices:
            if port_str.endswith(device):
                return port_str.split(" ")[0]

    # attempt to pick one with a device attached
    for port in ports:
        port_str = str(port)
        if not port_str.endswith("n/a"):
            return port_str.split(" ")[0]
    print("")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print("------------------------!!!!LIKELY ERROR!!!!---------------------")
    print(
        "------------------------!!!!VERIFY USB SERIAL IS PLUGGED IN!!!!---------------------"
    )
    print(
        "Didnt find a port with a connected FTDI device from the list:",
        "[" + ",".join(ftdi_devices) + "]",
        "picking the first port instead.",
        file=sys.stderr,
    )
    print("")
    try:
        port = str(ports[0]).split(" ")[0]
    except IndexError:
        ic("No serial ports were found. Exiting.")
        sys.exit(1)
    return port


@attr.s(auto_attribs=True)
class SerialQueue:
    # https://pyserial.readthedocs.io/en/latest/pyserial_api.html
    rx_queue: Queue
    tx_queue: Queue
    serial_data_dir: Path
    log_serial_data: bool
    ready_signal: str
    serial_port: str
    baud_rate: int = 460800
    default_timeout: float = 1.0
    hardware_buffer_size: int = 4096

    def listen_serial(self):
        if not self.serial_port:
            ic("No serial port specified, picking:")
            self.serial_port = pick_serial_port()

        ic(self.serial_port)
        serial_data_dir = self.serial_data_dir / Path("serial_logs")
        serial_data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_int_timestamp()
        serial_data_file = serial_data_dir / Path(
            timestamp + "_" + self.serial_port.split("/")[-1]
        )

        # https://pyserial.readthedocs.io/en/latest/pyserial_api.html#serial.serial_for_url
        # https://pyserial.readthedocs.io/en/latest/url_handlers.html#urls
        serial_url_list = ["spy://", self.serial_port]

        serial_url_list.append("?file=")
        if self.log_serial_data:
            serial_url_list.append(serial_data_file.as_posix())
        else:
            if sys.platform == "linux":
                serial_url_list.append("/dev/null")
            else:
                serial_url_list.append("NUL:")

        if self.log_serial_data:
            icp(self.serial_data_dir)
        serial_url = "".join(serial_url_list)
        icp(serial_url)
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
            self.ser,
        )
        # ic(self.ser.read_until)
        # ic(self.ser.nonblocking)
        if gvd:
            ic(dir(self.ser))
        discard = self.ser.readall()  # self.ser.readlines() is incorrect
        if gvd:
            ic(discard)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.rx_queue.put([self.ready_signal])
        while True:
            # if gvd:
            #    ic(self.ser.inWaiting())
            # bytes_buffered = self.ser.inWaiting()
            # if bytes_buffered == self.hardware_buffer_size:
            #    msg = "serial hardware overflow at bytes_buffered: {bytes_buffered} and self.rx_queue.qsize()"
            #    msg = msg.format(bytes_buffered, self.rx_queue.qsize()))
            #    raise ValueError(msg)
            # if bytes_buffered > (0.90 * self.hardware_buffer_size):
            #    msg = "WARNING: bytes_buffered: {} is > 90% of hardware_buffer_size: {}".format(bytes_buffered, self.hardware_buffer_size)
            #    print(msg, file=sys.stderr)
            read_bytes = self.ser.read(self.ser.inWaiting())
            if len(read_bytes) == self.hardware_buffer_size:
                msg = "serial hardware overflow at hardware_buffer_size: {} and self.rx_queue.qsize: {}"
                msg = msg.format(self.hardware_buffer_size, self.rx_queue.qsize())
                ic(read_bytes)
                raise ValueError(msg)

            # loops constantly
            # if gvd:
            #    ic(read_bytes)
            if read_bytes:
                self.rx_queue.put([read_bytes])

            if self.ser.inWaiting() == 0:
                while self.tx_queue.qsize() > 0:
                    ic(self.tx_queue.qsize())
                    try:
                        _exit_on_list = self.tx_queue.get(False)
                        if _exit_on_list == ["EXIT"]:
                            icp("got [EXIT]")
                            # raise KeyboardInterrupt
                            sys.exit(0)
                        _tx_data = _exit_on_list[0]
                        # icp(_tx_data)
                        if gvd:
                            ic(_tx_data)
                        _bytes_written = self.ser.write(_tx_data)
                        assert _bytes_written == len(_tx_data)
                        self.ser.flush()
                        if gvd:
                            ic("wrote:", _tx_data)
                    except Empty as e:  # oddness.
                        if gvd:
                            ic(e)

            # if self.ser.inWaiting() == 0:
            #    time.sleep(0.1)


def launch_serial_queue_process(
    *,
    rx_queue: Queue,
    tx_queue: Queue,
    serial_port: str,
    serial_data_dir: Path,
    baud_rate: int,
    log_serial_data: bool,
):
    ready_signal = str(time.time())
    serial_queue = SerialQueue(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_port=serial_port,
        baud_rate=baud_rate,
        serial_data_dir=serial_data_dir,
        log_serial_data=log_serial_data,
        ready_signal=ready_signal,
    )
    serial_queue_process = Process(target=serial_queue.listen_serial, args=())
    serial_queue_process.start()
    while True:
        try:
            ready_signal_response = rx_queue.get(False)[0]
        except Empty:
            continue
        ic(ready_signal_response)
        if ready_signal_response == ready_signal:
            break
        raise ValueError(ready_signal_response, ready_signal)  # testme
    return serial_queue_process


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


def read_fifo(io_handle, length: int) -> None | bytes:
    try:
        buffer: None | bytes = os.read(io_handle, length)
    except OSError as err:
        if err.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
            buffer = None
        else:
            raise

    if buffer in {None, b""}:
        pass
    else:
        if gvd:
            eprint(f"read_fifo() {len(buffer)=} {buffer=}")

    return buffer


def print_serial_output(
    *,
    serial_port: str | None,
    serial_data_dir: Path,
    log_serial_data: bool,
    timestamp: bool,
    read_tx_from_fifo: bool,
    baud_rate: int = 460800,
    show_bytes: bool = False,
):
    rx_queue = Queue()
    tx_queue = Queue()
    tx_tosend_queue = []
    last_queue_size = None
    queue_size = 0
    fifo_handle: int | None = None
    if read_tx_from_fifo:
        fifo_handle = os.open("/delme/fifo", os.O_RDONLY | os.O_NONBLOCK)
    serial_queue_process = launch_serial_queue_process(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_port=serial_port,
        baud_rate=baud_rate,
        serial_data_dir=serial_data_dir,
        log_serial_data=log_serial_data,
    )
    while True:
        if gvd:
            queue_size = rx_queue.qsize()
            if queue_size != last_queue_size:
                ic(queue_size)
                last_queue_size = queue_size
        try:
            # icp("try")
            data = rx_queue.get(False)
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
        if read_tx_from_fifo:
            _result = read_fifo(io_handle=fifo_handle, length=32)
            if _result:
                ic(_result)
                tx_queue.put([_result])
        # except Exception as e:
        #    ic(e)
        #    ic(type(e))


@attr.s(auto_attribs=True)
class SerialOracle:
    baud_rate: int
    serial_data_dir: Path
    log_serial_data: bool
    serial_port: str
    ipython_on_communication_error: bool

    def __attrs_post_init__(self):
        self.rx_queue = Queue()
        self.tx_queue = Queue()
        self.rx_buffer = bytearray()
        self.rx_buffer_cursor = 0
        self.serial_queue_process = launch_serial_queue_process(
            rx_queue=self.rx_queue,
            tx_queue=self.tx_queue,
            serial_port=self.serial_port,
            log_serial_data=self.log_serial_data,
            baud_rate=self.baud_rate,
            serial_data_dir=self.serial_data_dir,
        )

    def terminate(self):
        self.serial_queue_process.terminate()
        self.serial_queue_process.kill()
        self.serial_queue_process.close()

    def status(self):
        ic(self.rx_queue.qsize())
        ic(len(self.rx_buffer))
        ic(self.rx_buffer_bytes_available())

    def rx_buffer_bytes_available(self):
        result = len(self.rx_buffer) - self.rx_buffer_cursor
        return result

    def reset_rx(self):
        self.rx_buffer = bytearray()
        self.rx_buffer_cursor = 0
        try:
            while True:
                self.rx_queue.get(False)[0]  # raises Empty
        except Empty as e:
            pass

    # accepts a float but only inf
    def _read(self, *, count: int | float, progress: bool = False):
        if isinstance(count, float):
            assert count == inf
        while self.rx_buffer_bytes_available() < count:
            try:
                data = self.rx_queue.get(False)[0]  # raises Empty
                self.rx_buffer.extend(data)
            except Empty as e:
                if count != inf:
                    raise e
                if gvd:
                    ic("got expected exception Empty, breaking")
                break
            if progress:
                _len = len(self.rx_buffer)
                eprint(
                    f"{count}/{_len} {_len-count}     {int(_len/count*100)}%       ",
                    end="\r",
                )
        # icp("exiting while")

        if count != inf:
            result = self.rx_buffer[
                self.rx_buffer_cursor : self.rx_buffer_cursor + count
            ]
        else:
            result = self.rx_buffer[self.rx_buffer_cursor :]
        if gvd:
            ic(result)

        self.rx_buffer_cursor += len(result)
        return result

    def write(self, data: bytes) -> None:
        ic(data)
        self.tx_queue.put([data])

    def send_serial_command(
        self,
        command: bytes,
        expect_ack: bool,
        argument: None | bytes = None,
        data_bytes_expected: int = 0,
        byte_count_requested: bool | int = False,  # a spectific number of bytes
        bytes_expected=None,
        timeout: None | float = None,
        no_read: bool = False,
        echo: bool = True,
        simulate: bool = False,
    ):
        if simulate or gvd:
            echo = True
        ic(command, argument, expect_ack, timeout)
        assert isinstance(command, bytes)

        if data_bytes_expected:
            assert not byte_count_requested
            assert byte_count_requested != inf
            assert not bytes_expected
            assert not no_read

        if expect_ack:
            assert not byte_count_requested
            assert not bytes_expected
            assert not no_read
            assert byte_count_requested != inf

        if byte_count_requested:
            assert not no_read

        if no_read:
            assert not byte_count_requested
            assert byte_count_requested != inf
        elif byte_count_requested == inf:
            assert not no_read

        _command = construct_serial_command(command=command, argument=argument)

        if echo:
            _argument_repr = repr(argument)[0:10]
            eprint(
                "serialtool: send_serial_command()",
                f"{len(command)=}",
                f"{_command=}",
                f"{expect_ack=}",
                f"argument={_argument_repr}",
                f"{data_bytes_expected=}",
                f"{byte_count_requested=}",
                f"{bytes_expected=}",
                f"{timeout=}",
                f"{no_read=}",
                f"{_command.hex()=}",
            )

        ic(
            _command,
            len(_command),
            expect_ack,
            argument,
            byte_count_requested,
            bytes_expected,
            data_bytes_expected,
            timeout,
            no_read,
        )

        if simulate:
            return b""

        self.reset_rx()
        self.write(_command)

        if data_bytes_expected:
            assert not byte_count_requested
            # b"\x10\x02" + data_bytes_expected + b"\x10\x03"
            byte_count_requested = 2 + data_bytes_expected + 2

        if expect_ack:
            # in py, "False + 3 = 3" since "False == 0" is True
            byte_count_requested = byte_count_requested + 3

        # icp(byte_count_requested)
        if not no_read:
            # if expect_ack:
            #    if not timeout:
            #        timeout = 1
            try:
                rx_bytes = self.read_command_result(
                    byte_count_requested=byte_count_requested,
                    bytes_expected=bytes_expected,
                    timeout=timeout,
                )
            except ValueError as e:
                ic(e)
                raise e

            rx_bytes = self.extract_command_result(
                two_byte_command=command,
                result=rx_bytes,
                expect_ack=expect_ack,
                data_bytes_expected=data_bytes_expected,
            )

            if echo:
                if len(rx_bytes) > 100:
                    if gvd:
                        eprint(
                            f"serialtool: send_serial_command() {len(rx_bytes)=} {repr(rx_bytes)=}"
                        )
                    else:
                        eprint(
                            f"serialtool: send_serial_command() {len(rx_bytes)=} {repr(rx_bytes[:100])=}"
                        )
                else:
                    eprint(
                        f"serialtool: send_serial_command() {len(rx_bytes)=} {repr(rx_bytes)=}"
                    )
            return rx_bytes

    def send_serial_command_direct(
        self,
        command: bytes,
        byte_count_requested: None | int = None,
        expected_response: None | bytes = None,
        timeout: None | float = None,
        echo: bool = True,
    ):
        ic(command, expected_response, timeout)
        assert isinstance(command, bytes)

        if expected_response:
            assert byte_count_requested != inf
            assert not byte_count_requested  # will be calculated instead

        if byte_count_requested:
            assert not expected_response

        # _byte_count_requested = 0
        # if expected_response:
        #    _byte_count_requested = len(expected_response)

        if echo:
            eprint(
                "serialtool: send_serial_command_direct()",
                f"{command=}",
                f"{len(command)=}",
                f"{command.hex()=}",
                f"{timeout=}",
                f"{byte_count_requested=}",
                end="",
            )
            if expected_response:
                eprint(f" {expected_response=}", end="")  # deliberate sp
            eprint()

        ic(
            command,
            len(command),
            byte_count_requested,
            expected_response,
            timeout,
        )

        self.reset_rx()
        self.write(command)

        try:
            rx_bytes = self.read_command_result(
                byte_count_requested=byte_count_requested,
                bytes_expected=expected_response,
                timeout=timeout,
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
        ic(
            byte_count_requested,
            bytes_expected,
            timeout,
        )
        if bytes_expected:
            assert isinstance(bytes_expected, bytes)

        eprint(
            f"serialtool: read_command_result() {byte_count_requested=}, {bytes_expected=}, {timeout=}"
        )

        # better to force non-specificaion of the count in the calling code
        # if bytes_expected and byte_count_requested:
        #    assert len(bytes_expected) == byte_count_requested

        # dont both specify the expected response bytes AND the count of thouse same bytes
        if bytes_expected:
            assert byte_count_requested is None
            byte_count_requested = len(bytes_expected)

        if not timeout:
            timeout = inf
            ic(timeout)

        if byte_count_requested == 0:
            assert bytes_expected is None
            bytes_expected = b""
            assert self.rx_queue.qsize() == 0
            return

        if byte_count_requested == inf:  # could be 0, not raising NoResponseError
            assert False
            assert timeout > 0
            assert bytes_expected is None
            all_bytes = b""
            start_time = time.time()
            ic(start_time, timeout)
            while True:
                read_bytes = self._read(count=inf)  # aka byte_count_requested
                if gvd and read_bytes:
                    ic(read_bytes)
                all_bytes += read_bytes
                if (time.time() - start_time) > timeout:
                    ic("TIMEOUT", timeout)
                    raise TimeoutError(timeout)
            ic(len(all_bytes))
            return all_bytes

        result = b""
        start_time = time.time()
        ic(start_time, timeout, byte_count_requested)

        while len(result) < byte_count_requested:
            bytes_needed = byte_count_requested - len(result)

            if bytes_needed > 0:
                try:
                    result += self._read(count=bytes_needed, progress=progress)
                except Empty as e:
                    # ic("got exception Empty:", e)
                    pass  # the timeout will break loop

            if (time.time() - start_time) > timeout:
                ic("TIMEOUT", timeout)
                raise TimeoutError(timeout)
        if progress:
            eprint(f"\ndone: {byte_count_requested}/{len(result)}")
        _duration = time.time() - start_time
        if _duration > 0.25:
            _bytes_per_second = int(len(result) / _duration)
            _bits_per_second = _bytes_per_second * 8
            eprint(f"{_duration=}, {_bytes_per_second=}, {_bits_per_second=}")

        if gvd:
            ic(repr(result))  # all data
        ic(len(result), byte_count_requested)
        if byte_count_requested > 10:
            if gvd:
                icp(result[-10:])

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
        eprint(
            f"serialtool: extract_command_result() {two_byte_command=} {result=} {expect_ack=} {data_bytes_expected=}"
        )
        assert isinstance(two_byte_command, bytes)
        if expect_ack:
            ending_bytes_expected = b"\x06" + two_byte_command
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
            ic(result[0:2])
            assert result[0:2] == b"\x10\x02"
            result = result[2:]
            assert result[-2:] == b"\x10\x03"
            result = result[:-2]
        ic(result)
        return result


@click.command()
@click.argument("serial_port", type=str, nargs=1)
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
@click.option("--baud-rate", type=int, default=460800)
@click.option("--timestamp", is_flag=True)
@click.option("--read-from-fifo", is_flag=True)
@click.option("--log-serial-data", is_flag=True)
@click_add_options(click_global_options)
@click.pass_context
def cli(
    ctx,
    serial_port: str,
    data_dir: Path,
    show_bytes: bool,
    log_serial_data: bool,
    baud_rate: int,
    read_from_fifo: bool,
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
    )

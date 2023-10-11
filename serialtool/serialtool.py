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
from cycloidal_client.command_dict import COMMAND_DICT
from cycloidal_client.exceptions import SerialNoResponseError
from eprint import eprint
from globalverbose import gvd
from serial.tools import list_ports
from timestamptool import get_int_timestamp
from timestamptool import get_timestamp

# from contextlib import ExitStack
# from shutil import get_terminal_size

# import subprocess
# import atexit


DATA_DIR = Path(Path(os.path.expanduser("~")) / Path(".cycloidal_client"))
DATA_DIR.mkdir(exist_ok=True)


def generate_serial_port_help():
    help_text = "Available serial ports: "
    ports = list_ports.comports()
    _ports = [str(port) for port in ports]
    help_text = repr(tuple(_ports))
    # for port in ports:
    #    help_text += "\b\n" + str(port)

    help_text.replace("\n\n", "\n")
    return help_text


def lookup_two_byte_command_name(two_bytes: bytes):
    assert isinstance(two_bytes, bytes)
    two_bytes_str = two_bytes.split(b"0x")[-1].decode("utf8")
    for key, value in COMMAND_DICT.items():
        # value = COMMAND_DICT[key]
        value_hex_str = value.split("0x")[-1]
        value_bytes = bytearray.fromhex(value_hex_str)
        value_bytes_str = value_bytes.decode("utf8")
        if two_bytes_str == value_bytes_str:
            return key
    raise ValueError(two_bytes)


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
    default_timeout: float = 0.7
    hardware_buffer_size: int = 4096
    verbose: bool | int | float = False

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
        serial_url = ["spy://", self.serial_port]

        serial_url.append("?file=")
        if self.log_serial_data:
            serial_url.append(serial_data_file.as_posix())
        else:
            if sys.platform == "linux":
                serial_url.append("/dev/null")
            else:
                serial_url.append("NUL:")

        if self.log_serial_data:
            icp(self.serial_data_dir)
        serial_url = "".join(serial_url)
        icp(serial_url)
        self.ser = serial.serial_for_url(serial_url)
        ic(self.ser.port)
        self.ser.baudrate = self.baud_rate
        ic(self.ser.baudrate)
        self.ser.timeout = self.default_timeout
        ic(
            self.ser.timeout,
            self.ser.parity,
            self.ser.bytesize,
            self.ser.interCharTimeout,
            self.ser.inter_byte_timeout,
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
                    if self.verbose:
                        ic(self.tx_queue.qsize())
                    try:
                        data = self.tx_queue.get(False)[0]
                        if gvd:
                            ic(data)
                        self.ser.write(data)
                        if gvd:
                            ic("wrote:", data)
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
    verbose: bool = False,
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
    show_bytes: bool = False,
    verbose: bool = False,
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
            # else:
            byte_count_written_to_stdout = sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if gvd:
                ic(byte_count_written_to_stdout)
        except Empty:
            pass
        except Exception as e:
            ic(e)
            ic(type(e))


def print_serial_output(
    *,
    serial_port: str | None,
    serial_data_dir: Path,
    log_serial_data: bool,
    timestamp: bool,
    baud_rate: int = 460800,
    show_bytes: bool = False,
    verbose: bool = False,
):
    rx_queue = Queue()
    tx_queue = Queue()
    last_queue_size = None
    queue_size = 0
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
            data = rx_queue.get(False)
            data = data[0]
            if show_bytes:
                ic(data)
            if timestamp:
                _timestamp = get_timestamp()
                data = _timestamp.encode("utf8") + b" " + data
                if gvd:
                    ic(data)
            # else:
            byte_count_written_to_stdout = sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if gvd:
                ic(byte_count_written_to_stdout)
        except Empty:
            pass
        except Exception as e:
            ic(e)
            ic(type(e))


@attr.s(auto_attribs=True)
class SerialOracle:
    baud_rate: int
    serial_data_dir: Path
    log_serial_data: bool
    serial_port: str
    ipython_on_communication_error: bool
    verbose: bool | int | float = False

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

    def status(self):
        ic(self.rx_queue.qsize())
        ic(len(self.rx_buffer))
        ic(self.bytes_available())

    def bytes_available(self):
        # this constantly loops
        # if gvd:
        #    ic(self.rx_queue.qsize())
        #    ic(len(self.rx_buffer))
        result = len(self.rx_buffer) - self.rx_buffer_cursor
        # if gvd:
        #    ic(result)
        return result

    def _read(self, count=inf):
        # this constantly loops
        # if gvd:
        #    ic(count, "entering while")
        while self.bytes_available() < count:
            try:
                data = self.rx_queue.get(False)[0]  # raises Empty
                self.rx_buffer.extend(data)
            except Empty as e:
                if count != inf:
                    raise e
                if gvd:
                    ic("got expected exception Empty, breaking")
                break
        if count != inf:
            result = self.rx_buffer[
                self.rx_buffer_cursor : self.rx_buffer_cursor + count
            ]
        else:
            result = self.rx_buffer[self.rx_buffer_cursor :]
        if gvd:
            ic(result)
        # self.rx_buffer_cursor += count
        self.rx_buffer_cursor += len(result)  # handle the inf case
        return result

    def write(self, data):
        if self.verbose:
            ic(data)
        self.tx_queue.put([data])

    def send_serial_command(
        self,
        command: bytes,
        expect_ack: bool,
        argument: None | bytes = None,
        byte_count_requested=False,
        bytes_expected=None,
        timeout: None | int = None,
        no_read: bool = False,
        verbose: bool = False,
        echo: bool = True,
    ):
        command_name = lookup_two_byte_command_name(two_bytes=command)
        ic(command, command_name, argument, expect_ack, timeout)
        if not isinstance(command, bytes):
            raise TypeError(f"type(command) must be bytes, not {type(command)}")
        if argument:
            if not isinstance(argument, bytes):
                raise TypeError(
                    f"type(argument) must be bytes, not {type(argument)}",
                )

        if expect_ack:
            assert not byte_count_requested
            byte_count_requested = 3
            assert not bytes_expected
            bytes_expected = b"\x06" + command
            assert not no_read
            assert byte_count_requested != inf

        if byte_count_requested:
            assert not no_read

        if no_read:
            assert not byte_count_requested
            assert byte_count_requested != inf
        elif byte_count_requested == inf:
            assert not no_read
        else:
            assert byte_count_requested

        if argument:
            command = command + argument

        command = b"\x10\x02" + command + b"\x10\x03"

        if echo:
            _argument_repr = repr(argument)[0:10]
            eprint(
                f"{timeout=}",
                f"{expect_ack=}",
                f"{len(command)=}",
                f"argument={_argument_repr}",
                command,
                f"{byte_count_requested=}",
                command.hex(),
            )

        if verbose:
            ic(
                command,
                len(command),
                expect_ack,
                byte_count_requested,
                bytes_expected,
                timeout,
            )

        self.write(command)
        if not no_read:
            if expect_ack:
                if not timeout:
                    timeout = 10
            try:
                rx_bytes = self.read_command_result(
                    byte_count_requested=byte_count_requested,
                    bytes_expected=bytes_expected,
                    timeout=timeout,
                )
            except ValueError as e:
                ic(e)
                raise e

            if expect_ack:
                if rx_bytes != bytes_expected:
                    ic(rx_bytes)
                    ic(bytes_expected)
                    raise ValueError(rx_bytes)
            if echo:
                eprint(f"{len(rx_bytes)=}")
                # if len(rx_bytes) < 10:
                eprint(f"{repr(rx_bytes)=}")
            return rx_bytes

    def read_command_result(
        self,
        *,
        byte_count_requested: int,
        bytes_expected: None | bytes = None,
        expect_empty: bool = False,
        timeout: None | float = None,
        verbose: bool = False,
        progress: bool = False,
    ):
        """
        Reads byte_count_requested number of bytes over serial and compares it to bytes_expected if given.
        This function performs the acknowledgement reponse checks.
        bytes_expected = None:
            no bytes expected to be read back
        bytes_expected = True
            byte_count_requested of arb bytes to be read back
        """
        if bytes_expected:
            assert isinstance(bytes_expected, bytes)

        # ic(byte_count_requested, bytes_expected, expect_empty, timeout)
        eprint(
            f"read_command_result() {byte_count_requested=}, {bytes_expected=}, {expect_empty=}, {timeout=}"
        )
        if not timeout:
            timeout = inf
            ic(timeout)

        if byte_count_requested == 0:
            assert expect_empty
            assert bytes_expected is None
            bytes_expected = b""
            assert self.rx_queue.qsize() == 0
            return

        if byte_count_requested == inf:  # could be 0, not raising NoResponseError
            assert not expect_empty
            assert timeout > 0
            assert bytes_expected is None
            all_bytes = b""
            start_time = time.time()
            ic(start_time, timeout)
            while True:
                read_bytes = self._read(inf)  # aka byte_count_requested
                if verbose and read_bytes:
                    ic(read_bytes)
                all_bytes += read_bytes
                if (time.time() - start_time) > timeout:
                    ic("TIMEOUT", timeout)
                    break
            ic(len(all_bytes))
            return all_bytes

        result = b""
        start_time = time.time()
        ic(start_time, timeout, byte_count_requested)

        while len(result) < byte_count_requested:
            bytes_needed = byte_count_requested - len(result)

            if progress:
                eprint(f"{byte_count_requested}/{len(result)}", end="\r")

            if bytes_needed > 0:
                try:
                    result += self._read(bytes_needed)
                except Empty as e:
                    # ic("got exception Empty:", e)
                    pass  # the timeout will break loop

            if (time.time() - start_time) > timeout:
                ic("TIMEOUT", timeout)
                break

        if progress:
            eprint(f"{byte_count_requested}/{len(result)}")

        if verbose == inf:
            ic(repr(result))  # all data
        ic(len(result))
        ic(byte_count_requested)

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

        if result.startswith(b"\x06"):
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
@click.option("--log-serial-data", is_flag=True)
@click_add_options(click_global_options)
def cli(
    serial_port: str,
    data_dir: Path,
    show_bytes: bool,
    log_serial_data: bool,
    baud_rate: int,
    timestamp: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
):
    if not verbose:
        ic.disable()

    if not serial_port:
        serial_port = pick_serial_port()
    # else:
    # serial_port = serial_port[0]

    print_serial_output(
        serial_port=serial_port,
        serial_data_dir=data_dir,
        log_serial_data=log_serial_data,
        timestamp=timestamp,
        show_bytes=show_bytes,
        baud_rate=baud_rate,
    )

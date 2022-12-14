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
from clicktool import click_add_options
from clicktool import click_global_options
from cycloidal_client.command_dict import COMMAND_DICT
from cycloidal_client.exceptions import SerialNoResponseError
from icecream import ic
from serial.tools import list_ports

# from contextlib import ExitStack
# from shutil import get_terminal_size

# import subprocess
# import atexit


DATA_DIR = Path(Path(os.path.expanduser("~")) / Path(".cycloidal_client"))
DATA_DIR.mkdir(exist_ok=True)


def get_timestamp(decimals=0):
    timestamp = str(f"%.{decimals}f" % time.time())
    return timestamp


def generate_serial_port_help():
    help_text = "Available serial ports: "
    ports = list_ports.comports()
    for port in ports:
        help_text += "\b\n" + str(port)

    help_text.replace("\n\n", "\n")
    return help_text


def lookup_two_byte_command_name(two_bytes: bytes):
    assert isinstance(two_bytes, bytes)
    two_bytes_str = two_bytes.split(b"0x")[-1].decode("utf8")
    for key in COMMAND_DICT.keys():
        value = COMMAND_DICT[key]
        value_hex_str = value.split("0x")[-1]
        value_bytes = bytearray.fromhex(value_hex_str)
        value_bytes_str = value_bytes.decode("utf8")
        if two_bytes_str == value_bytes_str:
            return key


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
    verbose: bool | int | float
    serial_data_dir: Path
    log_serial_data: bool
    ready_signal: str
    serial_port: str
    serial_baud_rate: int = 460800
    default_timeout: float = 0.7
    hardware_buffer_size = 4096

    def listen_serial(self):
        if not self.serial_port:
            ic("No serial port specified, picking:")
            self.serial_port = pick_serial_port()

        ic(self.serial_port)
        ic(self.serial_data_dir)
        serial_data_dir = self.serial_data_dir / Path("serial_logs")
        serial_data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = get_timestamp()
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

        serial_url = "".join(serial_url)
        ic(serial_url)
        self.ser = serial.serial_for_url(serial_url)
        ic(self.ser.port)
        self.ser.baudrate = self.serial_baud_rate
        ic(self.ser.baudrate)
        self.ser.timeout = self.default_timeout
        ic(self.ser.timeout)
        ic(self.ser.parity)
        ic(self.ser.bytesize)
        ic(self.ser.interCharTimeout)
        ic(self.ser.inter_byte_timeout)
        ic(self.ser)
        # ic(self.ser.read_until)
        # ic(self.ser.nonblocking)
        if self.verbose:
            ic(dir(self.ser))
        discard = self.ser.readall()  # self.ser.readlines() is incorrect
        if self.verbose == inf:
            ic(discard)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.rx_queue.put([self.ready_signal])
        while True:
            # if self.verbose == inf:
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

            if self.verbose == inf:
                ic(read_bytes)
            if read_bytes:
                self.rx_queue.put([read_bytes])

            if self.ser.inWaiting() == 0:
                while self.tx_queue.qsize() > 0:
                    if self.verbose:
                        ic(self.tx_queue.qsize())
                    try:
                        data = self.tx_queue.get(False)[0]
                        if self.verbose == inf:
                            ic(data)
                        self.ser.write(data)
                        if self.verbose:
                            ic("wrote:", data)
                    except Empty as e:  # oddness.
                        if self.verbose == inf:
                            ic(e)


def launch_serial_queue_process(
    *,
    rx_queue: Queue,
    tx_queue: Queue,
    serial_port: str,
    serial_data_dir: Path,
    serial_baud_rate: int,
    log_serial_data: bool,
    verbose: bool | int | float,
):
    ready_signal = str(time.time())
    serial_queue = SerialQueue(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_port=serial_port,
        serial_baud_rate=serial_baud_rate,
        serial_data_dir=serial_data_dir,
        log_serial_data=log_serial_data,
        ready_signal=ready_signal,
        verbose=verbose,
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


def print_serial_output(
    *,
    serial_port: str,
    serial_data_dir: Path,
    log_serial_data: bool,
    verbose: bool | int | float,
    serial_baud_rate: int = 460800,
    show_bytes: bool = False,
):
    rx_queue = Queue()
    tx_queue = Queue()
    last_queue_size = None
    queue_size = 0
    serial_queue_process = launch_serial_queue_process(
        rx_queue=rx_queue,
        tx_queue=tx_queue,
        serial_port=serial_port,
        serial_baud_rate=serial_baud_rate,
        serial_data_dir=serial_data_dir,
        log_serial_data=log_serial_data,
        verbose=verbose,
    )
    while True:
        if verbose == inf:
            queue_size = rx_queue.qsize()
            if queue_size != last_queue_size:
                ic(queue_size)
                last_queue_size = queue_size
        try:
            data = rx_queue.get(False)
            data = data[0]
            if show_bytes:
                ic(data)
            else:
                byte_count_written_to_stdout = sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                if verbose == inf:
                    ic(byte_count_written_to_stdout)
        except Empty:
            pass
        except Exception as e:
            if verbose:
                ic(e)
                ic(type(e))


@attr.s(auto_attribs=True)
class SerialOracle:
    verbose: bool | int | float
    serial_baud_rate: int
    serial_data_dir: Path
    log_serial_data: bool
    serial_port: str

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
            serial_baud_rate=self.serial_baud_rate,
            serial_data_dir=self.serial_data_dir,
            verbose=self.verbose,
        )

    def status(self):
        ic(self.rx_queue.qsize())
        ic(len(self.rx_buffer))
        ic(self.bytes_available())

    def bytes_available(self):
        if self.verbose == inf:
            ic(self.rx_queue.qsize())
            ic(len(self.rx_buffer))
        result = len(self.rx_buffer) - self.rx_buffer_cursor
        if self.verbose == inf:
            ic(result)
        return result

    def _read(self, count=inf):
        if self.verbose == inf:
            ic(count, "entering while")
        while self.bytes_available() < count:
            try:
                data = self.rx_queue.get(False)[0]  # raises Empty
                self.rx_buffer.extend(data)
            except Empty as e:
                if count != inf:
                    raise e
                if self.verbose == inf:
                    ic("got expected exception Empty, breaking")
                break
        if count != inf:
            result = self.rx_buffer[
                self.rx_buffer_cursor : self.rx_buffer_cursor + count
            ]
        else:
            result = self.rx_buffer[self.rx_buffer_cursor :]
        if self.verbose == inf:
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
        verbose: bool | int | float,
        argument: None | bytes = None,
        byte_count_requested=False,
        bytes_expected=None,
        expect_ack: bool = False,
        timeout: None | int = None,
        no_read: bool = False,
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
                    verbose=verbose,
                )
            except ValueError as e:
                ic(e)
                raise e

            if expect_ack:
                if rx_bytes != bytes_expected:
                    ic(rx_bytes)
                    ic(bytes_expected)
                    raise ValueError(rx_bytes)
            return rx_bytes

    def read_command_result(
        self,
        *,
        byte_count_requested: int,
        verbose: bool | int | float,
        bytes_expected: None | bytes = None,
        expect_empty: bool = False,
        timeout: None | float = None,
    ):
        """
        Reads byte_count_requested number of bytes over serial and compares it to bytes_expected if given.
        This function performs the acknowledgement reponse checks.
        bytes_expected = None:
            no bytes expected to be read back
        bytes_expected = True
            byte_count_requested of arb bytes to be read back
        """
        if verbose == inf:
            ic()
        if bytes_expected:
            assert isinstance(bytes_expected, bytes)

        if verbose:
            ic(byte_count_requested, bytes_expected, expect_empty, timeout)
        if not timeout:
            timeout = inf
            if verbose == inf:
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
            if verbose:
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
        if verbose:  # fixme
            ic(start_time, timeout, byte_count_requested)
        while len(result) < byte_count_requested:
            bytes_needed = byte_count_requested - len(result)

            if bytes_needed > 0:
                try:
                    result += self._read(bytes_needed)
                except Empty as e:
                    # ic("got exception Empty:", e)
                    pass  # the timeout will break loop

            if (time.time() - start_time) > timeout:
                ic("TIMEOUT", timeout)
                break

        if verbose == inf:
            ic(repr(result))
            ic(len(result))
            ic(byte_count_requested)

        if bytes_expected:
            if len(result) == 0:
                raise SerialNoResponseError()

            if result != bytes_expected:
                ic(byte_count_requested)
                ic()
                ic(repr(bytes_expected), len(bytes_expected))
                ic("    ", repr(result), len(result))
                import IPython

                IPython.embed()
                raise ValueError(result)

        if result.startswith(b"\x06"):
            ic(result)
        return result


@click.command()
@click.argument("serial_port", type=str, nargs=-1)
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
@click.option("--log-serial-data", is_flag=True)
@click_add_options(click_global_options)
def cli(
    serial_port: str | None,
    data_dir: Path,
    show_bytes: bool,
    log_serial_data: bool,
    verbose: bool | int | float,
    verbose_inf: bool,
    dict_output: bool,
):

    if not serial_port:
        serial_port = pick_serial_port()
    print_serial_output(
        serial_port=serial_port,
        serial_data_dir=data_dir,
        log_serial_data=log_serial_data,
        show_bytes=show_bytes,
        verbose=verbose,
    )

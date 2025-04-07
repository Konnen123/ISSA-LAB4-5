import socket
import select
import pickle
import datetime

from typing import *


class ObjectSocketParams:
    """Static configuration parameters for ObjectSocket communication."""

    OBJECT_HEADER_SIZE_BYTES = 4  # Number of bytes used to encode the object size
    DEFAULT_TIMEOUT_S = 1         # Default timeout in seconds for receiving operations
    CHUNK_SIZE_BYTES = 1024       # Number of bytes to read per chunk


class ObjectSenderSocket:
    """
    A server-side socket for sending Python objects to a connected receiver.

    Attributes:
        ip (str): IP address to bind the socket to.
        port (int): Port number to bind the socket to.
        sock (socket.socket): The main server socket.
        conn (socket.socket): The active connection socket to the receiver.
        print_when_awaiting_receiver (bool): Whether to print status while waiting for receiver.
        print_when_sending_object (bool): Whether to print status when sending an object.
    """

    def __init__(self, ip: str, port: int,
                 print_when_awaiting_receiver: bool = False,
                 print_when_sending_object: bool = False):
        """
        Initializes and binds the socket, then waits for a receiver to connect.

        Args:
            ip (str): IP address to bind.
            port (int): Port number to bind.
            print_when_awaiting_receiver (bool): Print logs while awaiting receiver.
            print_when_sending_object (bool): Print logs during object sending.
        """
        self.ip = ip
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.conn = None

        self.print_when_awaiting_receiver = print_when_awaiting_receiver
        self.print_when_sending_object = print_when_sending_object

        self.await_receiver_conection()

    def await_receiver_conection(self):
        """
        Listens for and accepts a single incoming receiver connection.
        """
        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] awaiting receiver connection...')

        self.sock.listen(1)
        self.conn, _ = self.sock.accept()

        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] receiver connected')

    def close(self):
        """
        Closes the active connection to the receiver.
        """
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """
        Checks whether a receiver is currently connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.conn is not None

    def send_object(self, obj: Any):
        """
        Serializes and sends a Python object to the connected receiver.

        The object is first serialized using `pickle`, then sent over the
        socket with a fixed-size header indicating the data length.

        Args:
            obj (Any): The Python object to send.
        """
        data = pickle.dumps(obj)
        data_size = len(data)
        data_size_encoded = data_size.to_bytes(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES, 'little')
        self.conn.sendall(data_size_encoded)
        self.conn.sendall(data)

        if self.print_when_sending_object:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] Sent object of size {data_size} bytes.')


class ObjectReceiverSocket:
    """
    A client-side socket for receiving Python objects sent from a sender.

    Attributes:
        ip (str): IP address of the sender.
        port (int): Port number of the sender.
        conn (socket.socket): Socket connected to the sender.
        print_when_connecting_to_sender (bool): Whether to print connection status.
        print_when_receiving_object (bool): Whether to print logs upon receiving objects.
    """

    def __init__(self, ip: str, port: int,
                 print_when_connecting_to_sender: bool = False,
                 print_when_receiving_object: bool = False):
        """
        Initializes and connects to the sender.

        Args:
            ip (str): IP address of the sender.
            port (int): Port number of the sender.
            print_when_connecting_to_sender (bool): Print logs during connection.
            print_when_receiving_object (bool): Print logs when receiving objects.
        """
        self.ip = ip
        self.port = port
        self.print_when_connecting_to_sender = print_when_connecting_to_sender
        self.print_when_receiving_object = print_when_receiving_object

        self.connect_to_sender()

    def connect_to_sender(self):
        """
        Connects to the sender using TCP.
        """
        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connecting to sender...')

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.ip, self.port))

        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connected to sender')

    def close(self):
        """
        Closes the connection to the sender.
        """
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """
        Checks whether the socket is currently connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.conn is not None

    def recv_object(self) -> Any:
        """
        Receives and deserializes a Python object from the sender.

        First reads the header to determine the size, then reads the complete
        serialized payload and decodes it with `pickle`.

        Returns:
            Any: The received Python object.
        """
        obj_size_bytes = self._recv_object_size()
        data = self._recv_all(obj_size_bytes)
        obj = pickle.loads(data)

        if self.print_when_receiving_object:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] Received object of size {obj_size_bytes} bytes.')

        return obj

    def _recv_with_timeout(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> Optional[bytes]:
        """
        Attempts to receive a chunk of data with a timeout.

        Args:
            n_bytes (int): Number of bytes to read.
            timeout_s (float): Timeout in seconds.

        Returns:
            Optional[bytes]: The data received, or None if timeout occurs.
        """
        rlist, _1, _2 = select.select([self.conn], [], [], timeout_s)
        if rlist:
            data = self.conn.recv(n_bytes)
            return data
        else:
            return None

    def _recv_all(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> bytes:
        """
        Receives exactly `n_bytes` bytes from the socket, handling partial reads and timeouts.

        Args:
            n_bytes (int): Total number of bytes to receive.
            timeout_s (float): Timeout in seconds for each read.

        Raises:
            socket.error: If a timeout occurs before receiving the full payload.

        Returns:
            bytes: The full data payload.
        """
        data = []
        left_to_recv = n_bytes
        while left_to_recv > 0:
            desired_chunk_size = min(ObjectSocketParams.CHUNK_SIZE_BYTES, left_to_recv)
            chunk = self._recv_with_timeout(desired_chunk_size, timeout_s)
            if chunk is not None:
                data += [chunk]
                left_to_recv -= len(chunk)
            else:
                bytes_received = sum(map(len, data))
                raise socket.error(f'Timeout elapsed without any new data being received. '
                                   f'{bytes_received} / {n_bytes} bytes received.')
        return b''.join(data)

    def _recv_object_size(self) -> int:
        """
        Reads and decodes the object size from the fixed-size header.

        Returns:
            int: The size of the upcoming serialized object in bytes.
        """
        data = self._recv_all(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES)
        return int.from_bytes(data, 'little')

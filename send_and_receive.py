import socket
import sys

INPUT_PORT = 5005
OUTPUT_PORT = 5006


def send_and_receive(query_text):
    if not query_text:
        print("[!] No query provided.")
        return

    print(f"[*] Opening listener on port {OUTPUT_PORT}...")
    receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    receiver.bind(('0.0.0.0', OUTPUT_PORT))
    receiver.listen(1)
    receiver.settimeout(None)

    try:
        print(f"[*] Sending query to neural server on port {INPUT_PORT}...")
        sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender.connect(('localhost', INPUT_PORT))
        sender.sendall(query_text.encode('utf-8'))
        sender.close()

        print(f"[*] Query [{query_text}] sent. Waiting for AI to process (this may take a minute)...")
        conn, addr = receiver.accept()
        chunks = []
        while True:
            data = conn.recv(4096)
            if not data:
                break
            chunks.append(data.decode('utf-8'))

        result = "".join(chunks)

        print("\n" + "=" * 50)
        print("Server Response:")
        print("-" * 50)
        print(result if result else "[!] Received empty response.")
        print("=" * 50)
        conn.close()

    except ConnectionRefusedError:
        print("[!] Error: Could not connect to port 5005. Is the server running?")
    except KeyboardInterrupt:
        print("\n[!] Connection cancelled by user.")
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")
    finally:
        receiver.close()


if __name__ == "__main__":
    if not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    elif len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "stego"

    send_and_receive(query)
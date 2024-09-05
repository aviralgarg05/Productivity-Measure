import asyncio
import websockets
import cv2
import base64
import numpy as np

# Function to handle incoming WebSocket messages
async def handle_client(websocket, path):
    print("Client connected!")
    try:
        async for message in websocket:
            # Decode the received base64 string
            frame_data = base64.b64decode(message)

            # Convert it to a numpy array and decode the JPEG image
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Display the frame
            if frame is not None:
                cv2.imshow('Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except websockets.ConnectionClosed:
        print("Connection closed.")
    finally:
        cv2.destroyAllWindows()

# WebSocket server setup
async def main():
    print("WebSocket server starting...")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # Run the server forever

if __name__ == "__main__":
    asyncio.run(main())

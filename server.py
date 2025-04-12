import json
import asyncio
import socket
from websockets.exceptions import ConnectionClosedOK
from websockets.asyncio.server import serve
from torch import nn, save, load
from torch.utils.data import DataLoader
from Predictor import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
import cv2


# handler coroutine
async def handler(websocket):
    while True:
        try:
            message = await websocket.recv()
        except ConnectionClosedOK: # prevents gunking up error logs
            continue
        print("-----Received Message-----")
        imageDict = json.loads(message)
        print(f"Image Width: {imageDict["width"]}")
        print(f"Image Height: {imageDict["height"]}")

        flatImage = imageDict["vector"]
        imageMatrix = np.array(flatImage).reshape(imageDict["height"], imageDict["width"])
        print(f"Image Matrix Shape: {imageMatrix.shape}")
        # Display the bitmap in grayscale
        bitmap = np.array(imageMatrix, dtype=np.uint8)
        bitmap = cv2.resize(bitmap, (28, 28))
        print(f"Bitmap Shape: {bitmap.shape}")
        plt.close('all')
        plt.imshow(bitmap, cmap='gray_r', vmin=0, vmax=255)
        plt.colorbar(label='Grayscale Value')  # Optional: Add a colorbar to show the scale
        plt.title('Grayscale Bitmap')
        plt.show()


# main "wait" loop
async def main():
    # hostname = socket.gethostname()
    # ipaddress = socket.gethostbyname(hostname)
    ipaddress = "localhost"
    port = 8002
    async with serve(handler, ipaddress, port) as server:
        print(f"Server started at {ipaddress}:{port}")
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())


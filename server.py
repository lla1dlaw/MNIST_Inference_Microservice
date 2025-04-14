import pretty_errors
import json
import asyncio
import socket
from websockets.exceptions import ConnectionClosedOK
from websockets.asyncio.server import serve
import torch
import numpy as np
import cv2


# handler coroutine
async def handler(websocket):
    while True:
        try:
            message = await websocket.recv()
        except ConnectionClosedOK: # prevents gunking up error logs
            continue

        # initiallize model
        model = torch.load(".\\models\\50-50-50.pt", weights_only=False)
        model.to(torch.device("cpu"))
        model.eval()

        print("-----Received Message-----")
        imageDict = json.loads(message)
        print(f"Image Width: {imageDict['width']}")
        print(f"Image Height: {imageDict['height']}")

        # reshape the image to a 2D array, condense image, and flatten
        flatImage = imageDict["vector"]
        imageMatrix = np.array(flatImage).reshape(imageDict["height"], imageDict["width"])
        bitmap = np.array(imageMatrix, dtype=np.uint8)
        bitmap = cv2.resize(bitmap, (28, 28))
        bitmap = bitmap.flatten()

        # get prediction
        preds = model(torch.tensor(bitmap, dtype=torch.float32)).data
        print(f"Preds Shape: {preds.shape}")
        pred = torch.argmax(preds).item()
        print(f"\nPrediced Number: {pred}")
        

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


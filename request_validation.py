"""Validation for drawings posted to the recognition endpoints.

The image pipeline sizes its work from the request, so an unchecked array can
tie up the CPU for a long time. Only a flat array of 28 x 28 grayscale pixels
is accepted, which is what every page in static/ sends.
"""

import json

from fastapi import HTTPException, Request

IMAGE_PIXELS = 28 * 28
MAX_PIXEL_VALUE = 255
# A compact array is about 3.2 KB at most ("255," per pixel). This leaves room
# for whitespace from other clients.
MAX_BODY_BYTES = 8192


def parse_image(body: bytes) -> list[int]:
    """Return the pixels in body, or raise HTTPException 400 if not a valid image."""
    try:
        image = json.loads(body)
    except (ValueError, RecursionError):
        raise HTTPException(status_code=400, detail="Body is not valid JSON.")

    if not isinstance(image, list) or len(image) != IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Expected an array of exactly {IMAGE_PIXELS} pixel values.",
        )
    for value in image:
        # type() rather than isinstance(), since bool is a subclass of int.
        if type(value) is not int or not 0 <= value <= MAX_PIXEL_VALUE:
            raise HTTPException(
                status_code=400,
                detail=f"Pixel values must be integers from 0 to {MAX_PIXEL_VALUE}.",
            )
    return image


async def read_image(request: Request) -> list[int]:
    """FastAPI dependency that reads and validates a posted drawing.

    The body is read in chunks and abandoned at the limit, because the
    Content-Length header is optional and can be absent on chunked requests.
    """
    if request.headers.get("content-type") != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json.")

    too_large = HTTPException(status_code=413, detail="Request body is too large.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise too_large

    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise too_large
    return parse_image(bytes(body))

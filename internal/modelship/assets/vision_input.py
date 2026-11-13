"""Bounded data-image chat contract; never fetch request-provided URLs or paths."""
import base64
import binascii
from io import BytesIO
from typing import Annotated, Literal
import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_DATA_URL = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 64


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=32768)


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=MAX_DATA_URL, pattern=r"^data:image/(png|jpeg);base64,[A-Za-z0-9+/]+={0,2}$")


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class VisionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=32768)] | Annotated[list[ContentPart], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def bounded_content(self):
        if isinstance(self.content, str):
            if not self.content or len(self.content) > 32768:
                raise ValueError("invalid text message")
        else:
            if self.role != "user":
                raise ValueError("image content is supported in user messages only")
            text = "".join(p.text for p in self.content if isinstance(p, TextPart))
            if not text.strip() or len(text.encode("utf-8")) > 32768:
                raise ValueError("image messages require bounded accompanying text")
        return self


def image_count(messages):
    return sum(isinstance(part, ImagePart) for message in messages
               if isinstance(message.content, list) for part in message.content)


def decode_image(url):
    # Lazy import keeps legacy causal-LM text deployments independent of Pillow.
    from PIL import Image, ImageOps, UnidentifiedImageError

    formats = {"data:image/png;base64,": "PNG", "data:image/jpeg;base64,": "JPEG"}
    prefix = next((value for value in formats if url.startswith(value)), None)
    if prefix is None or len(url) > MAX_DATA_URL:
        raise ValueError("use one PNG/JPEG data URL under 4 MiB; remote/file URLs are not fetched")
    encoded = url[len(prefix):]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("invalid image encoding") from None
    if not data or len(data) > MAX_IMAGE_BYTES or base64.b64encode(data).decode("ascii") != encoded:
        raise ValueError("invalid or oversized image encoding")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if image.format != formats[prefix] or width < 1 or height < 1 or max(width, height) > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("invalid image MIME/dimensions or pixel budget exceeded")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated images are not supported")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
                output = ImageOps.exif_transpose(image).convert("RGB")
                output.info.clear()
                return output
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ValueError("image could not be decoded safely") from None


def normalize_messages(messages):
    """Return text/template parts and decoded PIL objects, never URI/path strings."""
    if image_count(messages) > 1:
        raise ValueError("one image is supported across the conversation request")
    normalized, images = [], []
    try:
        for message in messages:
            if isinstance(message.content, str):
                content = message.content
            else:
                content = []
                for part in message.content:
                    if isinstance(part, TextPart):
                        content.append({"type": "text", "text": part.text})
                    else:
                        images.append(decode_image(part.image_url.url))
                        content.append({"type": "image"})
            normalized.append({"role": message.role, "content": content})
        return normalized, images
    except Exception:
        for image in images:
            image.close()
        raise


def chat_capabilities(enabled, max_tokens=0):
    chat = {"image_input": enabled, "streaming": False, "tools": False, "max_output_tokens": 1024}
    if enabled:
        chat.update(max_images=1, max_image_bytes=MAX_IMAGE_BYTES,
                    max_image_pixels=MAX_IMAGE_PIXELS, max_image_dimension=MAX_IMAGE_DIMENSION,
                    image_mime_types=["image/png", "image/jpeg"], max_image_tokens=max_tokens,
                    recommended_image_side=768)
    return chat

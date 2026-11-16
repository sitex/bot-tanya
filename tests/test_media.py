import base64
import json
import struct
import zlib
from io import BytesIO
from typing import ClassVar, Literal

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from tanya_bot.schemas import ChatRequest, ErrorResponse
from tests.conftest import Harness

pytestmark = pytest.mark.anyio


def png_data(width: int = 1, padding: int = 0) -> bytes:
    def chunk(tag: bytes, content: bytes) -> bytes:
        checksum = zlib.crc32(tag + content)
        return (
            struct.pack(">I", len(content))
            + tag
            + content
            + struct.pack(">I", checksum)
        )

    header = struct.pack(">IIBBBBB", width, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"tEXt", b"comment\0" + b"x" * padding)
        + chunk(b"IDAT", zlib.compress(b"\0" + b"\xff\0\0" * width))
        + chunk(b"IEND", b"")
    )


def image_part(data: bytes, index: int = 1) -> str:
    return json.dumps(
        {
            "message_index": index,
            "kind": "image",
            "file_name": "selected.png",
            "data_url": "data:image/png;base64," + base64.b64encode(data).decode(),
        }
    )


def jpeg_data() -> bytes:
    with Image.new("RGB", (2, 1), "red") as image, BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


def body(request: ChatRequest, parts: str) -> str:
    return (
        request.model_dump_json(exclude_none=True)[:-1]
        + ',"attachments":'
        + parts
        + "}"
    )


class WirePart(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    type: Literal["input_text", "input_image"]
    text: str | None = None
    image_url: str | None = None
    detail: str | None = None


class WireMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | tuple[WirePart, ...]


class MediaDelivery(BaseModel):
    input: tuple[WireMessage, ...]


async def test_real_media_reaches_responses_at_its_user_turn_and_replays(
    harness: Harness, chat_request: ChatRequest
) -> None:
    text = "Extracted PDF page: red bicycle. https://example.invalid/a"
    text_part = json.dumps(
        {"message_index": 1, "kind": "text", "file_name": "notes.pdf", "text": text}
    )
    image = image_part(png_data())
    request = body(chat_request, f"[{image},{text_part}]")
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request)
        replay = await client.post("/v1/chat", content=request)

    assert response.status_code == replay.status_code == 200
    assert replay.content == response.content
    assert len(harness.wire.calls) == 1
    delivered = MediaDelivery.model_validate_json(harness.wire.calls[0].content)
    assert delivered.input[0] == WireMessage(
        role="assistant", content=chat_request.messages[0].content
    )
    assert delivered.input[1] == WireMessage(
        role="user",
        content=(
            WirePart(type="input_text", text=chat_request.messages[1].content),
            WirePart(type="input_text", text="selected.png"),
            WirePart(
                type="input_image",
                image_url="data:image/png;base64,"
                + base64.b64encode(png_data()).decode(),
                detail="auto",
            ),
            WirePart(type="input_text", text="notes.pdf"),
            WirePart(type="input_text", text=text),
        ),
    )


@pytest.mark.parametrize(
    "part",
    [
        '{"message_index":-1,"kind":"text","file_name":"x","text":"content"}',
        '{"message_index":0,"kind":"text","file_name":"x","text":"content"}',
        '{"message_index":2,"kind":"text","file_name":"x","text":"content"}',
        '{"message_index":true,"kind":"text","file_name":"x","text":"content"}',
        '{"message_index":"1","kind":"text","file_name":"x","text":"content"}',
        '{"message_index":1,"kind":"video","file_name":"x","text":"content"}',
        '{"message_index":1,"kind":"text","file_name":"../x","text":"content"}',
        '{"message_index":1,"kind":"text","file_name":"x\\nsecret","text":"content"}',
        '{"message_index":1,"kind":"text","file_name":"x","text":" "}',
        '{"message_index":1,"kind":"text","file_name":"x","text":"content","data_url":"x"}',
        '{"message_index":1,"kind":"image","file_name":"x","data_url":"https://127.0.0.1/secret"}',
        '{"message_index":1,"kind":"image","file_name":"x","data_url":"data:image/svg+xml;base64,PHN2Zz4="}',
        '{"message_index":1,"kind":"image","file_name":"x","data_url":"data:image/png;base64,!!!!"}',
        '{"message_index":1,"kind":"image","file_name":"x","data_url":"data:image/png;base64,eA=="}',
        image_part(png_data()).replace("image/png", "image/jpeg"),
        image_part(png_data()).replace("base64,", "base64,\n").replace("\n", "\\n"),
        image_part(png_data(1281)),
        image_part(png_data()[:33] + png_data()[-12:]),
        image_part(png_data()[:-12]),
        image_part(png_data()[:-20] + b"broken!!" + png_data()[-12:]),
        image_part(b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x01\x00\x01\x01\xff\xd9").replace(
            "image/png", "image/jpeg"
        ),
        image_part(jpeg_data()[:-2]).replace("image/png", "image/jpeg"),
    ],
)
async def test_invalid_media_is_sanitized_before_any_provider_request(
    harness: Harness, chat_request: ChatRequest, part: str
) -> None:
    async with harness.client() as client:
        response = await client.post(
            "/v1/chat", content=body(chat_request, f"[{part}]")
        )

    assert response.status_code == 422
    assert (
        ErrorResponse.model_validate_json(response.content).detail.code
        == "invalid_request"
    )
    assert "secret" not in response.text
    assert harness.wire.calls == []


@pytest.mark.parametrize(
    ("parts", "status"),
    [
        ("[" + image_part(jpeg_data()).replace("image/png", "image/jpeg") + "]", 200),
        (f"[{image_part(png_data(padding=120_000))}]", 200),
        (f"[{image_part(png_data(padding=1_048_576))}]", 413),
        ("[" + ",".join([image_part(png_data(padding=900_000))] * 5) + "]", 413),
        ("[" + ",".join([image_part(png_data())] * 9) + "]", 422),
        (
            json.dumps(
                [
                    {
                        "message_index": 1,
                        "kind": "text",
                        "file_name": "document",
                        "text": "x" * 16000,
                    }
                ]
            ),
            422,
        ),
        ("[]" + " " * (6 * 1_048_576), 413),
    ],
)
async def test_media_wire_and_context_budgets_are_enforced(
    harness: Harness, chat_request: ChatRequest, parts: str, status: int
) -> None:
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=body(chat_request, parts))

    assert response.status_code == status
    assert len(harness.wire.calls) == (1 if status == 200 else 0)


async def test_larger_body_requires_parsed_images_not_json_substrings(
    harness: Harness, chat_request: ChatRequest
) -> None:
    padded = body(chat_request, "[]") + " " * 128000
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=padded)

    assert response.status_code == 413
    assert harness.wire.calls == []


async def test_changed_media_cannot_reuse_a_successful_receipt(
    harness: Harness, chat_request: ChatRequest
) -> None:
    async with harness.client() as client:
        initial = await client.post(
            "/v1/chat", content=body(chat_request, f"[{image_part(png_data())}]")
        )
        changed = await client.post(
            "/v1/chat", content=body(chat_request, f"[{image_part(png_data(2))}]")
        )

    assert initial.status_code == 200
    assert changed.status_code == 409
    assert len(harness.wire.calls) == 1


@pytest.mark.parametrize(
    ("role", "has_media", "status"),
    [
        ("user", True, 200),
        ("user", False, 422),
        ("assistant", True, 422),
    ],
)
async def test_empty_text_requires_real_user_attachment_without_fake_message(
    harness: Harness,
    chat_request: ChatRequest,
    *,
    role: str,
    has_media: bool,
    status: int,
) -> None:
    request = body(chat_request, f"[{image_part(png_data())}]" if has_media else "[]")
    request = request.replace(
        '"role":"user","content":"I am learning to draw."',
        f'"role":"{role}","content":""',
    )
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request)

    assert response.status_code == status
    assert len(harness.wire.calls) == (1 if status == 200 else 0)
    if has_media and status == 200:
        delivered = MediaDelivery.model_validate_json(harness.wire.calls[0].content)
        assert delivered.input[1] == WireMessage(
            role="user",
            content=(
                WirePart(type="input_text", text="selected.png"),
                WirePart(
                    type="input_image",
                    image_url="data:image/png;base64,"
                    + base64.b64encode(png_data()).decode(),
                    detail="auto",
                ),
            ),
        )

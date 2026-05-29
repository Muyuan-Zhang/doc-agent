"""Unit tests for OpenAILLMClient."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.llm import OpenAILLMClient


def _connected_client() -> tuple[OpenAILLMClient, MagicMock]:
    client = OpenAILLMClient()
    mock_openai = MagicMock()
    client._client = mock_openai
    return client, mock_openai


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.data = [
        MagicMock(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    return resp


class TestOpenAILLMClientConnect:
    async def test_connect_sets_client(self):
        client = OpenAILLMClient()
        with patch("app.clients.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            with patch("app.clients.llm.AsyncOpenAI", return_value=MagicMock()):
                await client.connect()
        assert client._client is not None

    async def test_connect_passes_api_key(self):
        client = OpenAILLMClient()
        with patch("app.clients.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            with patch("app.clients.llm.AsyncOpenAI") as mock_cls:
                await client.connect()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("api_key") == "sk-test-key"

    async def test_connect_raises_when_api_key_missing(self):
        client = OpenAILLMClient()
        with patch("app.clients.llm.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                await client.connect()


class TestOpenAILLMClientDisconnect:
    async def test_disconnect_closes_client(self):
        client, mock_openai = _connected_client()
        mock_openai.close = AsyncMock()
        await client.disconnect()
        mock_openai.close.assert_awaited_once()

    async def test_disconnect_clears_client_reference(self):
        client, mock_openai = _connected_client()
        mock_openai.close = AsyncMock()
        await client.disconnect()
        assert client._client is None

    async def test_disconnect_is_idempotent(self):
        client = OpenAILLMClient()
        await client.disconnect()  # no error when _client is None


class TestOpenAILLMClientPing:
    async def test_ping_returns_true_when_embed_succeeds(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1]])
        )
        assert await client.ping() is True

    async def test_ping_returns_false_on_exception(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(side_effect=Exception("api error"))
        assert await client.ping() is False


class TestOpenAILLMClientEmbed:
    async def test_embed_returns_vector(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1, 0.2, 0.3]])
        )
        result = await client.embed("hello")
        assert result == [0.1, 0.2, 0.3]

    async def test_embed_calls_create_with_model(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1]])
        )
        await client.embed("test")
        _, kwargs = mock_openai.embeddings.create.call_args
        from app.core.config import settings
        assert kwargs["model"] == settings.openai_embedding_model

    async def test_embed_wraps_text_in_list(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1]])
        )
        await client.embed("hello")
        _, kwargs = mock_openai.embeddings.create.call_args
        assert kwargs["input"] == ["hello"]


class TestOpenAILLMClientEmbedBatch:
    async def test_empty_batch_returns_empty(self):
        client, _ = _connected_client()
        result = await client.embed_batch([])
        assert result == []

    async def test_batch_returns_all_vectors(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1], [0.2], [0.3]])
        )
        result = await client.embed_batch(["a", "b", "c"])
        assert len(result) == 3

    async def test_batch_calls_one_api_request(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1], [0.2]])
        )
        await client.embed_batch(["a", "b"])
        assert mock_openai.embeddings.create.await_count == 1

    async def test_batch_sorted_by_index(self):
        client, mock_openai = _connected_client()
        # Return data in reverse order
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.2], index=1), MagicMock(embedding=[0.1], index=0)]
        mock_openai.embeddings.create = AsyncMock(return_value=resp)
        result = await client.embed_batch(["first", "second"])
        assert result[0] == [0.1]
        assert result[1] == [0.2]

    async def test_batch_passes_all_texts(self):
        client, mock_openai = _connected_client()
        mock_openai.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([[0.1], [0.2]])
        )
        await client.embed_batch(["hello", "world"])
        _, kwargs = mock_openai.embeddings.create.call_args
        assert kwargs["input"] == ["hello", "world"]


class TestOpenAILLMClientComplete:
    async def test_complete_returns_string(self):
        client, mock_openai = _connected_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "The answer."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await client.complete("What is 2+2?")
        assert result == "The answer."

    async def test_complete_passes_max_tokens(self):
        client, mock_openai = _connected_client()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)
        await client.complete("prompt", max_tokens=256)
        _, kwargs = mock_openai.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 256

    async def test_complete_empty_content_returns_empty_string(self):
        client, mock_openai = _connected_client()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await client.complete("prompt")
        assert result == ""


class TestStreamComplete:
    def _make_chunk(self, content):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    def _make_stream_mock(self, contents):
        chunks = [self._make_chunk(c) for c in contents]

        async def _stream():
            for chunk in chunks:
                yield chunk

        return _stream()

    async def test_yields_tokens_from_stream(self):
        client, mock_openai = _connected_client()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_stream_mock(["Hello", " world"])
        )

        tokens = []
        async for token in client.stream_complete("test prompt"):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_skips_none_and_empty_delta_content(self):
        client, mock_openai = _connected_client()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_stream_mock(["token", None, "", "end"])
        )

        tokens = []
        async for token in client.stream_complete("prompt"):
            tokens.append(token)

        assert tokens == ["token", "end"]

    async def test_passes_stream_true_to_api(self):
        client, mock_openai = _connected_client()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()

        async def _empty():
            return
            yield  # noqa: unreachable — makes it an async generator

        mock_openai.chat.completions.create = AsyncMock(return_value=_empty())

        async for _ in client.stream_complete("prompt"):
            pass

        _, kwargs = mock_openai.chat.completions.create.call_args
        assert kwargs.get("stream") is True

    async def test_passes_max_tokens_kwarg(self):
        client, mock_openai = _connected_client()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()

        async def _empty():
            return
            yield  # noqa: unreachable

        mock_openai.chat.completions.create = AsyncMock(return_value=_empty())

        async for _ in client.stream_complete("prompt", max_tokens=256):
            pass

        _, kwargs = mock_openai.chat.completions.create.call_args
        assert kwargs.get("max_tokens") == 256

    async def test_yields_nothing_on_empty_stream(self):
        client, mock_openai = _connected_client()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            return_value=self._make_stream_mock([])
        )

        tokens = []
        async for token in client.stream_complete("prompt"):
            tokens.append(token)

        assert tokens == []

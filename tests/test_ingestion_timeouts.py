from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.ingestion as ingestion


class WhisperTranscriptionRetryTests(unittest.TestCase):
    def _make_temp_audio_file(self) -> Path:
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(fd, b"RIFF\x00\x00\x00\x00WAVE")
        finally:
            os.close(fd)
        return Path(path)

    @patch("app.ingestion.log_api_usage")
    @patch("app.ingestion.openai_client.audio.transcriptions.create")
    def test_whisper_retries_then_succeeds(self, mock_create, _mock_log_usage) -> None:
        audio_path = self._make_temp_audio_file()
        try:
            mock_create.side_effect = [
                Exception("transient timeout"),
                {
                    "text": "Recovered transcript",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.2,
                            "text": "Recovered transcript",
                        }
                    ],
                },
            ]

            with patch.object(ingestion, "WHISPER_TRANSCRIPTION_MAX_RETRIES", 2):
                payload = ingestion._transcribe_whisper_file(
                    audio_path=audio_path,
                    user_id="retry-user",
                    segment_offset_seconds=0.0,
                )

            self.assertEqual(mock_create.call_count, 2)
            self.assertEqual(payload.text, "Recovered transcript")
            for call in mock_create.call_args_list:
                self.assertIn("timeout", call.kwargs)
                self.assertEqual(call.kwargs["timeout"], ingestion.WHISPER_TRANSCRIPTION_TIMEOUT_SECONDS)
        finally:
            audio_path.unlink(missing_ok=True)

    @patch("app.ingestion.log_api_usage")
    @patch("app.ingestion.openai_client.audio.transcriptions.create")
    def test_whisper_raises_after_retry_budget(self, mock_create, _mock_log_usage) -> None:
        audio_path = self._make_temp_audio_file()
        try:
            mock_create.side_effect = Exception("persistent timeout")

            with patch.object(ingestion, "WHISPER_TRANSCRIPTION_MAX_RETRIES", 2):
                with self.assertRaises(RuntimeError) as exc_ctx:
                    ingestion._transcribe_whisper_file(
                        audio_path=audio_path,
                        user_id="retry-user",
                        segment_offset_seconds=0.0,
                    )

            self.assertIn("after 2 attempt(s)", str(exc_ctx.exception))
            self.assertEqual(mock_create.call_count, 2)
        finally:
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

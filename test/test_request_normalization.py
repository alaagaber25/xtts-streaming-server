import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemas.requests import StreamingInputs, TTSInputs  # noqa: E402


class RequestNormalizationTests(unittest.TestCase):
    def test_streaming_text_is_normalized(self):
        parsed_input = StreamingInputs(
            text="\u0623\u0647\u0644\u0627\u064b 123\u060c \u0645\u062f\u0631\u0633\u0629!",
            language="ar",
            speaker_profile_id="fahad",
        )

        self.assertNotIn("!", parsed_input.text)
        self.assertNotIn("\u060c", parsed_input.text)
        self.assertNotRegex(parsed_input.text, r"\d")
        self.assertIn("\u0645\u062f\u0631\u0633\u0647", parsed_input.text)

    def test_non_streaming_text_is_normalized(self):
        parsed_input = TTSInputs(
            text="  \u0627\u0644\u0633\u064e\u0651\u0644\u064e\u0627\u0645\u064f   \u0639\u0644\u064a\u0643\u0645 \u0664\u0662!  ",
            language="ar",
            speaker_profile_id="fahad",
        )

        self.assertNotIn("!", parsed_input.text)
        self.assertNotRegex(parsed_input.text, r"\d")
        self.assertIn("\u0627\u0644\u0633\u0644\u0627\u0645", parsed_input.text)


if __name__ == "__main__":
    unittest.main()

from unittest import TestCase
from urllib.parse import ParseResult
import avalan


class AvalanInitTestCase(TestCase):
    def test_license_and_site(self):
        self.assertEqual(avalan.license(), "MIT")
        site = avalan.site()
        self.assertIsInstance(site, ParseResult)
        self.assertEqual(site.scheme, "https")
        self.assertEqual(site.netloc, "avalan.ai")

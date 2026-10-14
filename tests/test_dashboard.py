"""A dashboard vékony megjelenítési rétegének tesztjei."""

import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import main


class TestDashboard(unittest.TestCase):
    """A riport megjelenítését és a kevés adat jelzését ellenőrzi."""

    def test_displays_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            path.write_text("<html>Test report</html>", encoding="utf-8")

            with (
                patch("dashboard.st") as streamlit,
                patch("dashboard.components") as components,
                patch("dashboard.build_drift_report") as build,
            ):
                build.return_value = (path, 3000, 30)
                main()

                streamlit.metric.assert_any_call("Referenciaautók", 3000)
                streamlit.metric.assert_any_call("Naplózott autók", 30)
                components.html.assert_called_once_with(
                    "<html>Test report</html>",
                    height=900,
                    scrolling=True,
                )

    def test_displays_information_when_data_is_insufficient(self) -> None:
        with (
            patch("dashboard.st") as streamlit,
            patch("dashboard.components") as components,
            patch(
                "dashboard.build_drift_report",
                side_effect=ValueError("Kevés adat"),
            ),
        ):
            main()
            streamlit.info.assert_called_once_with("Kevés adat")
            components.html.assert_not_called()

    def test_direct_execution(self) -> None:
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.caption"),
            patch("streamlit.button"),
            patch("streamlit.info") as information,
            patch(
                "src.monitoring.build_drift_report",
                side_effect=ValueError("Kevés adat"),
            ),
        ):
            runpy.run_path("dashboard.py", run_name="__main__")
            information.assert_called_once_with("Kevés adat")

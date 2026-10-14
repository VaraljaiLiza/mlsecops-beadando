"""Vékony Streamlit felület az Evidently-riport megjelenítésére."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.monitoring import build_drift_report


def main() -> None:
    """Megjeleníti az aktuális adatváltozási riportot."""
    st.set_page_config(page_title="CarDekho monitoring", layout="wide")
    st.title("CarDekho – Adatváltozás figyelése")
    st.caption(
        "Referencia: a projekt datasetje. "
        "Aktuális adatok: az API-n beküldött autók."
    )
    st.button("Riport frissítése")

    try:
        path, reference_count, current_count = build_drift_report(
            config_path=Path("params.yaml"),
            reference_path=Path("data/processed/cardekho_sample.csv"),
            events_path=Path("logs/events.jsonl"),
            output_path=Path("reports/data-drift.html"),
        )
    except (OSError, ValueError) as error:
        st.info(str(error))
        return

    st.metric("Referenciaautók", reference_count)
    st.metric("Naplózott autók", current_count)
    st.caption(
        "A drift az adatok eloszlásának változását jelzi; "
        "nem bizonyítja a modell pontosságának romlását."
    )
    components.html(
        path.read_text(encoding="utf-8"),
        height=900,
        scrolling=True,
    )


if __name__ == "__main__":
    main()

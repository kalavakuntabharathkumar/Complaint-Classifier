from src.prepare_data import clean_text

def test_clean_text_collapses_whitespace():
    assert clean_text("  hello\\n  world ") == "hello world"

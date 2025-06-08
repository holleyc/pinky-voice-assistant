# tests/test_lexical.py
import pytest
from PinkyWeb import extract_lexical_facts

@pytest.mark.parametrize("input_,expected", [
    ("My name is Alice",                     {"name": "Alice"}),
    ("call me Bob",                          {"name": "Bob"}),
    ("my age is 42",                         {"age": 42}),
    ("My favorite color is light blue",      {"favorite_color": "light blue"}),
    ("my favorite number is 7",              {"favorite_number": 7}),
    ("I drive a Tesla Model S",              {"vehicle": "Tesla Model S"}),
    # combinations:
    ("My name is Zoe and my age is 30",      {"name": "Zoe", "age": 30}),
])
def test_extract_facts(input_, expected):
    got = extract_lexical_facts(input_)
    assert got == expected

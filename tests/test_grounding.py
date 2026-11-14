import pytest

from bible_rag.grounding import book_index, check_answer
from bible_rag.index import Hit

from .conftest import PASSAGES

HITS = [Hit(p, 0.5) for passages in PASSAGES.values() for p in passages]


@pytest.mark.parametrize(
    "name,index",
    [
        ("John", 42),
        ("Juan", 42),
        ("1 Cor.", 45),
        ("1 Corintios", 45),
        ("Gen", 0),
        ("Génesis", 0),
        ("Psalms", 18),
        ("Salmo", 18),
        ("Song of Solomon", 21),
        ("Cantar de los Cantares", 21),
        ("Apocalipsis", 65),
        ("Chapter", None),
        ("J", None),
    ],
)
def test_book_index(name, index):
    assert book_index(name) == index


def test_references_in_both_languages_are_verified():
    report = check_answer("As John 3:16 says (Juan 3:17, RVR1960); see also Salmo 23:1.", HITS)
    assert [(c.text, c.supported) for c in report.references] == [
        ("John 3:16", True),
        ("Juan 3:17", True),
        ("Salmo 23:1", True),
    ]
    assert report.references[0].passage == "John 3:16-17 (KJV)"
    assert report.grounded


def test_unretrieved_references_and_invented_quotes_are_flagged():
    report = check_answer(
        'Paul says "charity suffereth long" (1 Cor. 13:4) and John 3:20; '
        "but also “Porque de tal manera amó Dios al mundo” (Juan 3:16).",
        HITS,
    )
    assert [(c.text, c.supported) for c in report.references] == [
        ("1 Cor. 13:4", False),
        ("John 3:20", False),
        ("Juan 3:16", True),
    ]
    assert [(c.text, c.supported) for c in report.quotes] == [
        ("charity suffereth long", False),
        ("Porque de tal manera amó Dios al mundo", True),
    ]
    assert not report.grounded and len(report.unsupported) == 3


def test_quotes_ignore_case_accents_punctuation_and_ellipsis():
    report = check_answer('"the LORD is my shepherd, I shall not want ... green pastures"', HITS)
    assert report.quotes[0].supported
    assert check_answer("“JEHOVA ES MI PASTOR; nada me faltara”", HITS).grounded


def test_answer_without_citations_is_not_grounded():
    report = check_answer("God is love.", HITS)
    assert not report.grounded and report.unsupported == []


def test_quoted_reference_is_checked_as_a_reference():
    report = check_answer('The answer is "John 3:16-17" and "Genesis 9:1".', HITS)
    assert report.quotes == []
    assert [(c.text, c.supported) for c in report.references] == [
        ("John 3:16-17", True),
        ("Genesis 9:1", False),
    ]


def test_short_quotes_do_not_shift_quote_pairing():
    answer = (
        'The word "rea" can refer to a friend or a neighbour, and the word "pastor" '
        "appears in “Jehová es mi pastor; nada me faltará”."
    )
    report = check_answer(answer, HITS)
    assert [(c.text, c.supported) for c in report.quotes] == [
        ("Jehová es mi pastor; nada me faltará", True)
    ]

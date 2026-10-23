from tripmate.kb.wikivoyage import chunk_text, parse_page

WIKITEXT = """{{pagebanner|x.jpg}}
'''Testville''' is a town in [[India]].
==Understand==
A [[small town|small]] place with&nbsp;history. [[File:pic.jpg|thumb|A caption]]
==See==
* {{see
| name=Old Fort | alt= | url=http://example.com
| lat=15.5 | long=73.9
| hours=9AM-5PM | price=₹50
| lastedit=2024-01-02
| content=A '''17th-century''' fort with [[Goa|sea]] views.
}}
* {{listing | type=do | name=River Cruise | content=One hour cruise.}}
* {{listing | type=other | name=Bank}}
* {{see | name= | content=nameless}}
==Stay safe==
Beware of touts.
{{geo|15.5|73.9}}
"""


def test_parse_page_extracts_listings_sections_and_geo():
    page = parse_page("Testville", WIKITEXT)
    assert page.geo == (15.5, 73.9)
    assert [(p["kind"], p["name"]) for p in page.listings] == [
        ("see", "Old Fort"),
        ("do", "River Cruise"),
    ]
    fort = page.listings[0]
    assert fort["hours"] == "9AM-5PM" and fort["price"] == "₹50"
    assert fort["lat"] == 15.5 and fort["lastedit"] == "2024-01-02"
    assert fort["content"] == "A 17th-century fort with sea views."
    sections = dict(page.sections)
    assert set(sections) == {"Understand", "Stay safe"}
    assert sections["Understand"] == "A small place with history."


def test_chunk_text_respects_limit_and_keeps_everything():
    text = "\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, max_chars=400)
    assert all(len(c) <= 400 for c in chunks)
    assert sum(c.count("Paragraph") for c in chunks) == 10

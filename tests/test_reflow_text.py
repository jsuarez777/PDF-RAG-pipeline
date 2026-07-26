"""Resilience tests for the pdfplumber text reflow, in conjunction with pysbd.

pdfplumber emits one visual line per '\\n' with no paragraph markers, so a
sentence wrapped across the page margin looks like many "sentences" to pysbd
and words broken at the margin keep a soft hyphen ('gener-\\nation'). reflow_text
(via reflow_lines) rebuilds paragraph-aware prose from positioned lines before
chunking. These tests feed reflow_lines synthetic page layouts and then run the
same pysbd segmenter the sentence chunkers use, asserting the reflow makes pysbd
see the *grammatical* sentences instead of layout fragments.

The headline case is chunk_436 from the AI_Agents plumber-struct run, which pysbd
shredded into 9 fragments from what is really 3 sentences (the 3rd was truncated
at the chunk edge; here it is completed as it appears on the page).
"""

import pysbd
import pytest

from app.pdfplumber_to_text import reflow_lines


def make_lines(paragraphs, line_h=8.0, line_gap=2.0, para_gap=14.0):
    """Build page_text_lines-style dicts from paragraphs of wrapped line strings.

    Each paragraph is a list of lines as they wrap on the page (hyphens and
    all). Lines within a paragraph sit line_gap apart; a new paragraph starts
    para_gap below the previous line, i.e. a blank-line-sized gap. The median
    line gap here is line_gap, so reflow_lines' paragraph threshold
    (median * 1.6 + 1) falls between line_gap and para_gap.
    """
    lines = []
    for para in paragraphs:
        for i, text in enumerate(para):
            top = 0.0 if not lines else lines[-1]["bottom"] + (para_gap if i == 0 else line_gap)
            lines.append({"text": text, "top": top, "bottom": top + line_h, "x0": 0.0, "x1": 100.0})
    return lines


@pytest.fixture(scope="module")
def seg():
    # Same configuration the sentence chunkers use (char_span for exact offsets).
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


def n_sentences(seg, text):
    return len(seg.segment(text))


# chunk_436: line-wrap hyphens (There-/fore, follow-/ing, Bill-/ing), a mid-
# sentence file extension (Moby-Dick / .txt), and an inline error-code blob with
# colons, braces and a bracketed ellipsis — every one of which pysbd mistook for
# a sentence boundary in the raw run.
CHUNK_436_LINES = [[
    "There-",
    "fore, consider further shortening the input text (in our case, the Moby-Dick",
    ".txt e-book file) if you want to limit expenses. Additionally, ensure your",
    "OpenAI API credit balance remains positive to avoid errors such as the follow-",
    "ing: RateLimitError: Error code: 429 - {'error': {'message': 'You",
    "exceeded your current quota, please check your plan and billing details",
    "[...].' If necessary, log in to the OpenAI API page, navigate to Settings > Bill-",
    "ing and add funds to your balance.",
]]


def test_chunk_436_reflows_to_three_sentences(seg):
    reflowed = reflow_lines(make_lines(CHUNK_436_LINES))

    # The three real sentences, each opening with its true first word.
    sents = [s.sent for s in seg.segment(reflowed)]
    assert len(sents) == 3
    assert sents[0].startswith("Therefore, consider")
    assert sents[1].startswith("Additionally, ensure")
    assert sents[2].startswith("If necessary, log in")

    # A single paragraph reflows to one continuous run — no stray newlines...
    assert "\n" not in reflowed
    # ...and every soft hyphen is healed into its whole word.
    assert "Therefore," in reflowed and "There- " not in reflowed
    assert "following:" in reflowed and "follow- " not in reflowed
    assert "Billing" in reflowed and "Bill- " not in reflowed


def test_reflow_beats_naive_newline_join(seg):
    """Regression guard: reflow must collapse the fragmentation, not preserve it."""
    naive = "\n".join(CHUNK_436_LINES[0])   # what raw extract_text() produced
    reflowed = reflow_lines(make_lines(CHUNK_436_LINES))

    naive_n = n_sentences(seg, naive)
    assert naive_n >= 7          # raw run was badly shredded (was 9)
    assert n_sentences(seg, reflowed) == 3
    assert n_sentences(seg, reflowed) < naive_n


def test_dehyphenates_wraps_but_keeps_real_compounds(seg):
    # Every line-end hyphen is a soft wrap; the mid-line hyphen in
    # "Retrieval-augmented" is a real compound and must survive.
    lines = [[
        "Retrieval-augmented gener-",
        "ation combines a retriev-",
        "er with a language model to ground answers in doc-",
        "uments.",
    ]]
    reflowed = reflow_lines(make_lines(lines))
    assert reflowed == (
        "Retrieval-augmented generation combines a retriever with a "
        "language model to ground answers in documents."
    )
    assert n_sentences(seg, reflowed) == 1


def test_number_range_and_abbreviations_are_not_mangled(seg):
    # A hyphen after a digit ('2010-') is not a word wrap, so the range must not
    # be fused into '20102012'; decimals ('3.5') and abbreviations ('U.S.') must
    # not trip pysbd into extra splits. 'inter-/national' is a genuine wrap.
    lines = [[
        "The reference period ran from 2010-",
        "2012. Growth in the U.S. reached 3.5 percent, and inter-",
        "national demand rose too.",
    ]]
    reflowed = reflow_lines(make_lines(lines))
    assert "2010" in reflowed and "2012" in reflowed
    assert "20102012" not in reflowed          # digit-hyphen was NOT joined
    assert "international" in reflowed          # letter-hyphen WAS joined
    assert n_sentences(seg, reflowed) == 2


def test_layout_gap_emits_paragraph_break():
    # A blank-line-sized vertical gap between the heading and the body becomes a
    # '\n\n' break; wraps within the multi-line body stay joined. (This is
    # reflow's job; whether pysbd then honors the break is separate — see below.)
    # The body must span several lines so the small within-paragraph gaps set the
    # baseline the heading gap stands out against, as on a real page.
    lines = [
        ["Introduction"],
        [
            "The study examines housing attitudes across",
            "the country. It surveys many house-",
            "holds across many regions of interest.",
        ],
    ]
    reflowed = reflow_lines(make_lines(lines))
    assert reflowed == (
        "Introduction\n\n"
        "The study examines housing attitudes across the country. "
        "It surveys many households across many regions of interest."
    )


def test_paragraph_break_keeps_heading_separate(seg):
    """A reconstructed paragraph break stops a heading merging into the body.

    pysbd treats a newline as a sentence boundary but a bare space as ordinary
    whitespace. So a punctuation-less heading ('Introduction') only stays its own
    unit if reflow puts a real break before the body — which is exactly why the
    layout gap is reconstructed as '\\n\\n' rather than collapsed to a space.
    """
    reflowed = reflow_lines(make_lines([
        ["Introduction"],
        [
            "The study examines housing attitudes across",
            "the country. It surveys many house-",
            "holds across many regions of interest.",
        ],
    ]))
    assert "\n\n" in reflowed
    sents = [s.sent.strip() for s in seg.segment(reflowed)]
    assert sents[0] == "Introduction"                     # heading is its own unit
    assert sents[1].startswith("The study examines")
    assert len(sents) == 3


def test_multi_paragraph_sentences_all_segment(seg):
    reflowed = reflow_lines(make_lines([
        ["The first paragraph has two sentences. Here is", "the second one, wrapped over two lines."],
        ["A new paragraph begins here. And it also", "ends here after wrapping."],
    ]))
    assert reflowed.count("\n\n") == 1
    assert n_sentences(seg, reflowed) == 4


def test_empty_and_single_line_inputs():
    assert reflow_lines([]) == ""
    one = make_lines([["Just one line of prose."]])
    assert reflow_lines(one) == "Just one line of prose."

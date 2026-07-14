"""
Fixture generator: a small but DIRTY corpus that reproduces the corpus-level
pathologies that hurt retrieval most. All in-memory, no external files.

Every doc is a `Doc(doc_id, filename, text, date)` where `date` is the doc's
own effective/last-modified date (ISO). The pathologies deliberately overlap the
way a real SharePoint/Drive dump does:

  NEAR-DUPLICATES / VERSIONING  refund_policy v1 / v2 / final / FINAL2
  CONTRADICTION                 old policy says 15 days, new says 30 days
  TEMPORAL VALIDITY             pricing_2023 vs pricing_2026 (both "current")
  SKEWED LENGTHS                a 2-line memo vs a ~400-line report
  PII IN FREE TEXT              Aadhaar (valid Verhoeff), PAN, account, card
  VOCAB MISMATCH                corpus says "pay statement"; user asks "salary slip"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Doc:
    doc_id: str
    filename: str
    text: str
    date: str  # ISO effective / last-modified date


# --- versioned near-duplicates (same policy, drifting wording) --------------
_REFUND_BASE = (
    "Refund Policy. Customers may request a refund for eligible orders. "
    "Refunds are processed to the original payment method within {window} days "
    "of an approved request. Shipping charges are non-refundable. "
    "Digital goods are refundable only if unused."
)


def _versioned() -> list[Doc]:
    return [
        Doc("d_v1", "refund_policy_v1.txt",
            _REFUND_BASE.format(window="15"), "2022-01-10"),
        Doc("d_v2", "refund_policy_v2.txt",
            _REFUND_BASE.format(window="15") + " Approved requests are reviewed by support.",
            "2023-03-01"),
        Doc("d_final", "refund_policy_final.txt",
            _REFUND_BASE.format(window="30") + " Approved requests are reviewed by support.",
            "2025-06-15"),
        Doc("d_final2", "refund_policy_FINAL2.txt",
            _REFUND_BASE.format(window="30") + " Approved requests are reviewed by support. "
            "This supersedes all prior refund policies.",
            "2026-02-20"),
    ]


# --- temporal pricing pair (both phrased as "current") ----------------------
def _pricing() -> list[Doc]:
    return [
        Doc("d_price23", "pricing_current.txt",
            "Current Pricing. Our Pro plan is priced at INR 999 per month, "
            "effective for the 2023 fiscal year.", "2023-04-01"),
        Doc("d_price26", "pricing_2026.txt",
            "Current Pricing. Our Pro plan is priced at INR 1499 per month, "
            "effective 1 April 2026.", "2026-04-01"),
    ]


# --- skewed lengths ---------------------------------------------------------
def _skewed() -> list[Doc]:
    memo = Doc("d_memo", "office_memo.txt",
               "Office closed on 15 Aug for the public holiday. Normal hours resume 16 Aug.",
               "2026-08-01")
    para = (
        "The ingestion subsystem processes documents through detection, "
        "normalization, and chunking stages. Each stage records provenance so "
        "downstream retrieval can attribute every chunk to a source. ")
    report = Doc("d_report", "annual_report.txt", (para * 120).strip(), "2026-01-05")
    return [memo, report]


# --- PII in free text (valid Aadhaar Verhoeff = 234123412346) ---------------
def _pii() -> list[Doc]:
    return [
        Doc("d_pii", "onboarding_note.txt",
            "Employee Rakesh submitted KYC. Aadhaar 2341 2341 2346, PAN ABCDE1234F. "
            "Salary credited to account 12345678901 at HDFC. "
            "Backup card 4111 1111 1111 1111 on file. Also a fake-looking 0000 0000 0000.",
            "2026-05-10"),
    ]


# --- vocab mismatch (corpus term vs user term) ------------------------------
def _vocab() -> list[Doc]:
    return [
        Doc("d_paystmt", "hr_pay_statement.txt",
            "How to download your pay statement. Log in to the HR portal, open "
            "Compensation, and export the monthly pay statement as PDF. The pay "
            "statement lists earnings, deductions, and net remuneration.",
            "2026-03-12"),
        Doc("d_leave", "hr_leave_policy.txt",
            "Leave policy. Employees accrue paid leave monthly. Apply via the HR "
            "portal at least three days in advance.", "2026-03-12"),
    ]


def make_corpus() -> list[Doc]:
    docs: list[Doc] = []
    docs += _versioned()
    docs += _pricing()
    docs += _skewed()
    docs += _pii()
    docs += _vocab()
    return docs


# The canonical vocab-mismatch probe: user's word never appears in the corpus.
VOCAB_QUERY = "how do I get my salary slip"


if __name__ == "__main__":
    for d in make_corpus():
        print(f"{d.doc_id:12s} {d.filename:26s} {d.date}  len={len(d.text):5d}  "
              f"{d.text[:50]!r}")

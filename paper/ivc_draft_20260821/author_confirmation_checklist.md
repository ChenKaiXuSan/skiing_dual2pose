# IVC pre-submission checklist

Updated 2026-09-18. Source of requirements: IVC Guide for Authors (verified against the 2024-03 archived copy; re-check the live page once logged in to Editorial Manager, especially the word limit).

## 1. Items that need input from the authors

All statements below are drafted in `main.tex` (marked with `% CONFIRM` comments) and summarized in `coauthor_confirmation_request.md`, which can be sent as-is.

- [ ] **Declaration of competing interest**: drafted as "no known competing interests"; every author replies; then generate the shared .docx at https://declarations.elsevier.com/ and upload it unchanged.
- [ ] **CRediT roles**: drafted for all six authors (Funding acquisition follows the KAKEN records: JP25K03146 PI Kitahara, co-I Xie; JP24K02978 co-I Kitahara); each author confirms their roles.
- [ ] **Data and code availability**: drafted (code public on GitHub; Unity data and checkpoint to be deposited with a DOI upon publication; author-collected videos not shared). Confirm release scope and choose the data repository (Zenodo/Mendeley Data).
- [ ] **Ethics statement**: drafted (consenting adult volunteers from the group; no institutional review required). Confirm who appears in the clips, that written consent covers publication, and the university's review/exemption position.
- [ ] **Generative-AI declaration**: drafted with Claude (Anthropic) and OpenAI Codex for readability and language. Confirm tool names and scope, or delete the section if no AI was used for writing.
- [ ] **Corresponding author ORCID**: candidate `0009-0003-8698-8899` (University of Tsukuba) found via ORCID search; confirm. Consider an institutional e-mail instead of chenkaixusan@gmail.com.
- [x] **Full postal addresses** written for all four affiliations (CCS 305-8577; Systems and Information Engineering 305-8573; CiNet, Suita 565-0871; NTT CS Labs, Atsugi 243-0198). Confirm wording with E. Naito (CiNet) and T. Kimura (Atsugi rather than Tokyo).
- [ ] **Author list**: Nan Luo removed on 2026-09-18; the list at submission is definitive, so confirm with all six remaining authors and with Nan Luo.
- [ ] Confirm final title and author order; confirm the manuscript is not under review elsewhere.
- [ ] Confirm acknowledgement text and funding identifiers (KAKENHI JP24K02978, JP25K03146; HPCI hp260146).
- [ ] Obtain the accepted MMSports 2026 manuscript PDF (ACM TAPS camera-ready) and place it in this directory for upload; confirm the camera-ready conference title.
- [ ] Add the MMSports DOI, ISBN and page range to references.bib and cover_letter.md once the ACM record is public.

## 2. Manuscript-level checks

- [ ] **Word count**: the archived guide states full-length papers must be 4500-7500 words excluding references; the current main text is about 9,500 words. Confirm whether the limit is still in force; if so, move material (e.g., Table 8, Figure 8, parts of Related Work) to the supplement.
- [x] Highlights: 5 bullets, 74-81 characters each, separate editable file named Highlights.txt.
- [x] Graphical abstract: graphical_abstract.pdf / .png, 13 x 5 cm, 1535 x 590 px (minimum 1328 x 531).
- [x] Keywords: 6 (journal maximum).
- [x] Abstract contains no citations.
- [x] Sections numbered; figures and tables have captions and are placed next to the text; supplementary items use S prefixes.
- [x] References: consistent numeric style (elsarticle-num); DOIs present where available. `wan2025rsb` author list corrected (added Yiming Bao).
- [x] All figure files below 10 MB.
- [x] Single-anonymized review: no anonymization needed.
- [x] Funding section present in Elsevier format.
- [ ] Final spell/grammar pass of main.pdf and supplementary.pdf.

## 3. Editorial Manager form fields

- [ ] Article type: Full Length Article.
- [ ] Suggested reviewers: enter the five people in suggested_reviewers.md (name, institution, institutional e-mail, reason).
- [ ] Colour: online only (free); no colour in print.
- [ ] Decide on the free SSRN preprint option and on open access vs. subscription.
- [ ] Do not upload main_marked.* or main_diff.*.

## 4. After acceptance

- Journal Publishing Agreement (or open-access licence) sent to the corresponding author.
- Editable source (LaTeX + figures) required at revision/acceptance.
- Proof corrections due within two days of receipt.

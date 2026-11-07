# Public chat API v1 contract copies (test-only)

The JSON files in this directory are **unchanged, byte-for-byte copies** of the
authoritative public chat API v1 contract. That contract lives in the separate
public `rastinaghighi.dev` repository, which remains authoritative: its schemas
and `docs/api/chat-v1.md` supersede the conceptual API draft in
backend-master-plan.md §7.2.

The copies let `tests/test_public_response.py` validate public responses and
requests with the standard `jsonschema` Draft 2020-12 validator when Mascot-Lab
is checked out and tested on its own. The test suite never reads the website
repository and needs no network access. Production code does not read this
directory.

Never edit a copy by hand. To take up a contract change, follow
[Updating the copies](#updating-the-copies).

## Record

Copied on 2026-11-07 from the working tree of the `rastinaghighi.dev` checkout
next to Mascot-Lab. Each copy was compared byte for byte with its original after
copying. All six originals were UTF-8 without a byte order mark, with LF line
endings and a final line feed. The record identifies the originals by path and
content hash only. No Git command was run, so it names no website commit.

| Copy | Original in rastinaghighi.dev | SHA-256 |
| --- | --- | --- |
| `chat-request-v1.schema.json` | `docs/api/schemas/chat-request-v1.schema.json` | `f01e14d356033b5cb3738a232e7e3dab396a43acb3c054b493502b0094b3bb57` |
| `chat-response-v1.schema.json` | `docs/api/schemas/chat-response-v1.schema.json` | `c02b908cc385073fe4db0d9d6f121e0c9103998c4cbbcac16d8a90eeeccdd367` |
| `examples/answered.json` | `docs/api/examples/answered.json` | `dc4de6ed7c956688888e390ff759e39125d5b7ecf1fbe2a9527a0b26c9e7a3e7` |
| `examples/error.json` | `docs/api/examples/error.json` | `6b14c1f50cb9d4ef0d69c3c4b26c96f932e0f1a524b137f1de523f2163e8cb26` |
| `examples/navigation.json` | `docs/api/examples/navigation.json` | `b435d9d00e412523562a80bce3d6fc3ab574cbd0a4b1d9883b93ebeca9ef66ee` |
| `examples/request.json` | `docs/api/examples/request.json` | `3d37741ae703dbeb6dc2ff5fdd785976f928c0e12f986eb8a81c05c7d802e874` |

The tests and `check_contract_drift.py` read this table, so keep its row format
exactly: three backquoted cells, and a lowercase hexadecimal SHA-256. Every JSON
file in this directory must have a row, and every row must name an existing
copy with the recorded SHA-256. Otherwise the tests fail.

Validator: `jsonschema` 4.26.0 (with `referencing` 0.37.0 and
`jsonschema-specifications` 2025.9.1) was installed when the copies were made.
`pyproject.toml` declares `jsonschema>=4.18` in the `dev` extra.

## What the tests check

`tests/test_public_response.py` loads both schemas when it is imported. A copy
that is missing, is not JSON, does not declare Draft 2020-12
(`https://json-schema.org/draft/2020-12/schema`) or fails
`Draft202012Validator.check_schema` makes the import fail. pytest then reports a
collection error and stops the run. Nothing is skipped. The schema check applies
the Draft 2020-12 format checker to the metaschema, so every `pattern` must be a
valid regular expression.

The tests also check that:

- every copy has its recorded SHA-256, and the set of JSON files matches this record;
- each schema uses only Draft 2020-12 keywords, so a misspelt keyword cannot
  silently weaken validation;
- the contract's examples conform, and every body the mapper may publish conforms
  to the response schema;
- representative valid and invalid responses and requests are accepted and
  rejected by the actual schemas;
- the drift check below reports identical, changed and missing originals correctly.

Instances are validated with `Draft202012Validator` and its `FORMAT_CHECKER`.
Neither schema uses the `format` keyword today, so the format checker has no
effect on instances. It applies to any `format` that a future version adds.

## Drift detection

When the website repository is available, compare the copies with their
originals:

```text
.\.venv\Scripts\python.exe tests\contracts\check_contract_drift.py [WEBSITE_REPOSITORY]
```

`WEBSITE_REPOSITORY` defaults to the `rastinaghighi.dev` checkout next to
Mascot-Lab. The script only reads files. For each recorded copy, it checks the
copy's SHA-256 against this record and compares the copy byte for byte with its
original. Its exit status follows `cmp` and `diff`:

| Status | Meaning |
| --- | --- |
| `0` | Every copy has its recorded SHA-256 and is identical to its original. |
| `1` | Drift: a copy differs from its recorded SHA-256 or from its original. The report gives both hashes and says when only line endings differ. |
| `2` | The check could not be completed: the record cannot be read, or the website repository, an original or a copy is missing. This takes precedence over `1`. |

The script compares only the recorded files. It does not report contract files
that the website repository adds later.

## Updating the copies

1. Review the contract change in `rastinaghighi.dev`. A contract change can
   require mapper or test changes, which are separate, owner-approved work.
2. Copy each changed original over its copy, unchanged.
3. Record the new SHA-256 values in the table above, for example with
   `Get-FileHash -Algorithm SHA256 <file>` or `sha256sum <file>` (lowercase
   the hexadecimal), and update the copy date.
4. Run the drift check, which must exit with status 0, and then the full test suite.

## Line endings

The SHA-256 values are of the exact bytes, with LF line endings. The
`.gitattributes` file in this directory turns off Git's line-ending conversion
for these JSON files, so a checkout with `core.autocrlf` enabled keeps them
identical to their originals. A website checkout that converts its own files to
CRLF makes the drift check report drift, noting that only line endings differ.

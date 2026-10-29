import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from crosshatetn import sources
from crosshatetn.sources import (SourceError, fetch, read_lhsab_tsv, read_thsab_xlsx, read_tunizi_txt, sha256_file,
                                 to_schema)

SPEC = {"role": "labeled", "dialect": "tunisian", "label_map": {"normal": "normal", "abusive": "abusive",
                                                                 "hate": "hate"}}


def test_tunizi_parsing_ignores_junk_prefix_but_keeps_minus_sign():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.txt"
        path.write_text('1;ana fil ghorba\n-1;mouch behi\n%1;merci\n"-1;ya3tik\n\n', encoding="utf-8")
        df = read_tunizi_txt(path)
    assert df["label_source"].tolist() == ["1", "-1", "1", "-1"]
    assert df["text_raw"].tolist() == ["ana fil ghorba", "mouch behi", "merci", "ya3tik"]


def test_tunizi_rejects_unreadable_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.txt"
        path.write_text("1;ok\nligne sans etiquette\n", encoding="utf-8")
        try:
            read_tunizi_txt(path)
        except SourceError:
            return
    raise AssertionError("une ligne illisible doit lever SourceError")


def test_lhsab_reads_quoted_multiline_tweets():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "L-HSAB"
        path.write_text('Tweet\tClass\n"سطر اول\n\nسطر ثاني"\tabusive\nتغريدة بسيطة\tnormal\n', encoding="utf-8")
        df = read_lhsab_tsv(path)
    assert len(df) == 2
    assert "\n" in df.loc[0, "text_raw"]
    assert df["label_source"].tolist() == ["abusive", "normal"]


def test_thsab_reads_excel_without_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "T-HSAB.xlsx"
        pd.DataFrame([["نص اول", "hate"], ["نص ثاني", "normal"]]).to_excel(path, header=False, index=False)
        df = read_thsab_xlsx(path)
    assert df.columns.tolist() == ["text_raw", "label_source"]
    assert len(df) == 2


def test_to_schema_builds_stable_ids_and_maps_labels():
    df = pd.DataFrame({"text_raw": ["a", None, "c"], "label_source": [" hate", "normal", "abusive"]})
    out = to_schema("thsab", SPEC, df)
    assert out["id"].tolist() == ["thsab-00001", "thsab-00002", "thsab-00003"]
    assert out["label"].tolist() == ["hate", "normal", "abusive"]
    assert out["text_raw"].tolist() == ["a", "", "c"]
    assert out["source_row"].tolist() == [1, 2, 3]


def test_to_schema_rejects_unknown_labels():
    df = pd.DataFrame({"text_raw": ["a"], "label_source": ["toxique"]})
    try:
        to_schema("thsab", SPEC, df)
    except SourceError:
        return
    raise AssertionError("une étiquette hors label_map doit lever SourceError")


def test_pool_sources_have_no_label():
    df = pd.DataFrame({"text_raw": ["sa7a"], "label_source": ["1"]})
    out = to_schema("tunizi", {"role": "pool", "dialect": "tunisian"}, df)
    assert out["label"].isna().all()
    assert out["label_source"].tolist() == ["1"]


def test_fetch_uses_verified_cache_and_rejects_altered_files():
    content = b"contenu de reference"
    spec = {"repo": "x/y", "commit": "abc", "path": "dossier/fichier.txt",
            "sha256": hashlib.sha256(content).hexdigest()}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        cached = raw / "src" / "fichier.txt"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(content)
        assert fetch("src", spec, raw) == cached          # cache valide : aucun accès réseau
        assert sha256_file(cached) == spec["sha256"]

        cached.write_bytes(b"fichier modifie")             # cache altéré : on retélécharge
        original = sources._download
        sources._download = lambda url, dest, timeout=60: Path(dest).write_bytes(b"autre version")
        try:
            fetch("src", spec, raw)
        except SourceError as exc:
            assert "Empreinte inattendue" in str(exc)
            assert not cached.exists()                    # le fichier non conforme est supprimé
        else:
            raise AssertionError("une empreinte différente doit lever SourceError")
        finally:
            sources._download = original

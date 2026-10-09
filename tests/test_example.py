"""Tests for downloading the example archives from the paramaterial-examples repository."""
import pytest
import requests

from paramaterial import example


def test_unknown_example_name_raises(tmp_path):
    with pytest.raises(ValueError, match='not recognized'):
        example.download_example('example_1', to_directory=str(tmp_path))


def _github_reachable() -> bool:
    try:
        return requests.head(example.BASE_URL, timeout=5).status_code < 500
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _github_reachable(), reason='needs network access to github.com')
def test_download_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the archive is downloaded to the working directory before extraction
    try:
        example.download_example('dan_msc_cs2_0.1.0', to_directory=str(tmp_path / 'out'))
    except Exception as e:
        if 'error occurred: 5' in str(e):  # a 5xx from GitHub is their outage, not our bug
            pytest.skip(str(e))
        raise
    notebook = tmp_path / 'out' / 'examples' / 'dan-msc-case-studies' / 'CS2' / 'dan-msc-cs2-0.1.0.ipynb'
    assert notebook.exists()
    assert not (tmp_path / 'dan_msc_cs2_0.1.0.tar.gz').exists()

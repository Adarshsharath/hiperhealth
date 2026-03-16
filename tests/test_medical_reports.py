"""Test the extraction of medical report data."""

import io
import os
import shutil

from pathlib import Path

import pytest

import hiperhealth.agents.extraction.medical_reports as medical_reports_mod
from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportExtractorError,
    MedicalReportFileExtractor,
    TextExtractionError,
)

TEST_DATA_PATH = Path(__file__).parent / 'data' / 'reports'
PDF_FILE = TEST_DATA_PATH / 'pdf_reports' / 'report-1.pdf'
IMAGE_FILE = TEST_DATA_PATH / 'image_reports' / 'image-1.png'
UNSUPPORTED_FILE = TEST_DATA_PATH / 'pdf_reports' / 'unsupported_file.txt'
CORRUPT_PDF_FILE = TEST_DATA_PATH / 'pdf_reports' / 'corrupt_report.txt'
HAS_TESSERACT = shutil.which('tesseract') is not None


@pytest.fixture
def extractor():
    """Return a MedicalReportFileExtractor instance for testing."""
    return MedicalReportFileExtractor()


def test_only_supported_files_can_be_extracted(extractor):
    """Test that only supported files can be validated successfully."""
    extractor._validate_or_raise(PDF_FILE)
    extractor._validate_or_raise(IMAGE_FILE)
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_text_from_pdf_file(extractor):
    """Test text extraction from PDF files returns valid string."""
    text = extractor._extract_text_from_pdf(PDF_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_text_from_image_uses_mocked_ocr(monkeypatch, extractor):
    """Image OCR should return mocked text without tesseract."""
    mocked_text = 'Mocked OCR text'

    monkeypatch.setattr(
        medical_reports_mod.pytesseract,
        'image_to_string',
        lambda _image: mocked_text,
    )

    text = extractor._extract_text_from_image(IMAGE_FILE)

    assert text == mocked_text


def test_extract_text_from_image_raises_when_mocked_ocr_empty(
    monkeypatch, extractor
):
    """Blank OCR output should raise a text extraction error."""
    monkeypatch.setattr(
        medical_reports_mod.pytesseract,
        'image_to_string',
        lambda _image: '   ',
    )

    with pytest.raises(TextExtractionError, match='No extractable text in image'):
        extractor._extract_text_from_image(IMAGE_FILE)


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_extract_text_from_image_file(extractor):
    """Test text extraction from image files using OCR."""
    text = extractor._extract_text_from_image(IMAGE_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_unsupported_file_raises(extractor):
    """Test that unsupported file types raise appropriate errors."""
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_corrupt_pdf_raises(extractor):
    """Test that corrupt PDF files raise TextExtractionError."""
    with pytest.raises(TextExtractionError):
        extractor._extract_text_from_pdf(CORRUPT_PDF_FILE)


def test_convert_to_fhir_uses_mocked_anamnesisai(monkeypatch, extractor):
    """FHIR conversion should serialize resources from mocked client."""
    Patient = type(
        'Patient',
        (),
        {
            'model_dump': lambda self: {
                'id': 'patient-1',
                'resourceType': 'Patient',
            }
        },
    )
    Observation = type(
        'Observation',
        (),
        {
            'model_dump': lambda self: {
                'id': 'obs-1',
                'resourceType': 'Observation',
            }
        },
    )

    captured = {}

    class FakeAnamnesisAI:
        def __init__(self, backend, api_key):
            captured['backend'] = backend
            captured['api_key'] = api_key

        def extract_fhir(self, text_content):
            captured['text_content'] = text_content
            return [[Patient(), Observation()]]

    monkeypatch.setattr(
        medical_reports_mod, 'AnamnesisAI', FakeAnamnesisAI
    )

    result = extractor._convert_to_fhir(
        'mock clinical text', api_key='test-key'
    )

    assert captured == {
        'backend': 'openai',
        'api_key': 'test-key',
        'text_content': 'mock clinical text',
    }
    assert result['Patient'] == {
        'id': 'patient-1',
        'resourceType': 'Patient',
    }
    assert result['Observation'] == {
        'id': 'obs-1',
        'resourceType': 'Observation',
    }


def test_convert_to_fhir_raises_when_api_key_missing(monkeypatch, extractor):
    """FHIR conversion should fail fast when no OpenAI API key is available."""
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    with pytest.raises(EnvironmentError, match='Missing OpenAI API key'):
        extractor._convert_to_fhir('anything', api_key=None)


def test_extract_report_data_orchestrates_validate_extract_convert(
    monkeypatch, extractor
):
    """Top-level extraction should validate, extract, and convert in order."""
    calls = []

    def fake_validate(source):
        calls.append(('validate', source))

    def fake_extract_text(source):
        calls.append(('extract', source))
        return 'mocked text'

    def fake_convert_to_fhir(text_content, api_key=None):
        calls.append(('convert', text_content, api_key))
        return {'Patient': {'id': '123'}}

    monkeypatch.setattr(extractor, '_validate_or_raise', fake_validate)
    monkeypatch.setattr(extractor, '_extract_text', fake_extract_text)
    monkeypatch.setattr(extractor, '_convert_to_fhir', fake_convert_to_fhir)

    result = extractor.extract_report_data(PDF_FILE, api_key='k')

    assert result == {'Patient': {'id': '123'}}
    assert calls == [
        ('validate', PDF_FILE),
        ('extract', PDF_FILE),
        ('convert', 'mocked text', 'k'),
    ]


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OpenAI API key not available'
)
def test_extract_report_data_from_pdf_file(extractor):
    """Test FHIR data extraction from PDF files."""
    api_key = os.environ.get('OPENAI_API_KEY')
    fhir_data = extractor.extract_report_data(PDF_FILE, api_key)
    assert isinstance(fhir_data, dict)
    assert len(fhir_data) > 0
    expected_keys = {'Patient', 'Condition', 'Observation', 'DiagnosticReport'}
    assert any(key in fhir_data for key in expected_keys)


@pytest.mark.skipif(
    (not os.environ.get('OPENAI_API_KEY')) or (not HAS_TESSERACT),
    reason='OpenAI API key or tesseract not available',
)
def test_extract_report_data_from_image_file(extractor):
    """Test FHIR data extraction from image files."""
    api_key = os.environ.get('OPENAI_API_KEY')
    fhir_data = extractor.extract_report_data(IMAGE_FILE, api_key)
    assert isinstance(fhir_data, dict)
    assert len(fhir_data) > 0
    expected_keys = {'Patient', 'Condition', 'Observation', 'DiagnosticReport'}
    assert any(key in fhir_data for key in expected_keys)


def test_extract_text_uses_cache(monkeypatch, extractor):
    """Repeated extraction should reuse cached text for the same source."""
    call_count = {'pdf': 0}

    monkeypatch.setattr(
        extractor, '_get_mime_type', lambda _source: 'application/pdf'
    )

    def fake_extract_text_from_pdf(source):
        call_count['pdf'] += 1
        assert source == PDF_FILE
        return 'cached pdf text'

    monkeypatch.setattr(
        extractor, '_extract_text_from_pdf', fake_extract_text_from_pdf
    )

    first = extractor._extract_text(PDF_FILE)
    second = extractor._extract_text(PDF_FILE)

    assert first == 'cached pdf text'
    assert second == 'cached pdf text'
    assert call_count['pdf'] == 1


def test_support_inmemory_pdf(extractor):
    """Test text extraction from in-memory PDF BytesIO objects."""
    with open(PDF_FILE, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_pdf(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_support_inmemory_image(extractor):
    """Test text extraction from in-memory image BytesIO objects."""
    with open(IMAGE_FILE, 'rb') as f:
        image_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_image(image_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


def test_empty_inmemory_file_raises(extractor):
    """Test that empty in-memory streams raise FileNotFoundError."""
    empty_stream = io.BytesIO(b'')
    with pytest.raises(FileNotFoundError):
        extractor._validate_or_raise(empty_stream)

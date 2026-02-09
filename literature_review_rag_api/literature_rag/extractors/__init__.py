"""PDF Extractors for Literature RAG

Provides different extraction strategies for various document types:
- AcademicPDFExtractor: Section-aware extraction for academic papers
- BusinessExtractor: Business document extraction (reports, memos)
- GenericExtractor: Simple full-text extraction

Usage:
    from literature_rag.extractors import create_extractor

    # Create specific extractor
    extractor = create_extractor("business")

    # Or auto-detect from PDF content
    extractor = create_extractor("auto", pdf_path=my_pdf_path)

    # Extract
    sections, metadata = extractor.extract_pdf(pdf_path)
"""

from .base import (
    BaseExtractor,
    GenericExtractor,
    BusinessExtractor,
    ExtractedSection,
    PDFMetadata,
    create_extractor,
)

__all__ = [
    "BaseExtractor",
    "GenericExtractor",
    "BusinessExtractor",
    "ExtractedSection",
    "PDFMetadata",
    "create_extractor",
]

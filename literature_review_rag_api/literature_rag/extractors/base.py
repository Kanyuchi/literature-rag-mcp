"""Base PDF Extractor Interface

Provides abstract base class for PDF extraction and a factory function
to create appropriate extractors based on document type.

Extractor types:
- "academic": Section-aware extraction for academic papers (abstract, methods, etc.)
- "business": Business document extraction (reports, memos, proposals)
- "generic": Simple full-text extraction without section detection
- "auto": Auto-detect based on document content

Usage:
    from literature_rag.extractors import create_extractor

    extractor = create_extractor("business")
    sections, metadata = extractor.extract_pdf(pdf_path)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """Represents a detected section in a PDF."""
    section_type: str
    content: str
    page_start: int
    page_end: int
    confidence: float


@dataclass
class PDFMetadata:
    """Rich metadata extracted from PDF."""
    # Document identification
    doc_id: str
    filename: str
    file_path: str

    # Bibliographic (may be empty for non-academic docs)
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None

    # Organization (from folder structure or user input)
    phase: Optional[str] = None
    phase_name: Optional[str] = None
    topic_category: Optional[str] = None

    # Content
    abstract: Optional[str] = None
    summary: Optional[str] = None  # For business docs
    keywords: Optional[List[str]] = None

    # Document stats
    total_pages: int = 0
    file_size_bytes: int = 0

    # Extraction stats
    sections_detected: int = 0
    extraction_method: str = "unknown"  # "section_aware", "full_text", "generic"
    document_type: str = "unknown"  # "academic", "business", "generic"


class BaseExtractor(ABC):
    """Abstract base class for PDF extraction.

    All extractors must implement:
    - extract_pdf(): Main extraction method returning sections and metadata
    - extract_full_text(): Fallback full-text extraction
    """

    def __init__(self, config: dict = None):
        """Initialize extractor with configuration.

        Args:
            config: Extraction configuration dictionary
        """
        self.config = config or {}
        self.max_pages = self.config.get("max_pages_per_pdf")

    @abstractmethod
    def extract_pdf(
        self,
        pdf_path: Path,
        phase_info: dict = None
    ) -> Tuple[Optional[List[ExtractedSection]], PDFMetadata]:
        """Extract PDF content and metadata.

        Args:
            pdf_path: Path to PDF file
            phase_info: Optional dict with phase, phase_name, topic_category

        Returns:
            (sections, metadata) tuple:
            - sections: List of ExtractedSection or None for full-text
            - metadata: PDFMetadata with extracted information
        """
        pass

    @abstractmethod
    def extract_full_text(self, pdf_path: Path) -> str:
        """Extract full text from PDF without section detection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Full text content as string
        """
        pass

    def _init_metadata(self, pdf_path: Path, phase_info: dict = None) -> PDFMetadata:
        """Initialize metadata structure with common fields."""
        phase_info = phase_info or {}
        doc_id = self._generate_doc_id(pdf_path, phase_info.get("phase"))

        return PDFMetadata(
            doc_id=doc_id,
            filename=pdf_path.name,
            file_path=str(pdf_path),
            phase=phase_info.get("phase"),
            phase_name=phase_info.get("phase_name"),
            topic_category=phase_info.get("topic_category"),
            file_size_bytes=pdf_path.stat().st_size if pdf_path.exists() else 0
        )

    def _generate_doc_id(self, pdf_path: Path, phase: str = None) -> str:
        """Generate unique document ID."""
        phase_prefix = phase.lower().replace(" ", "_") if phase else "doc"
        filename_part = pdf_path.stem[:50]
        filename_part = re.sub(r'[^a-z0-9_]', '_', filename_part.lower())
        return f"{phase_prefix}_{filename_part}"


class GenericExtractor(BaseExtractor):
    """Generic PDF extractor without section detection.

    Suitable for:
    - Business documents (reports, memos, proposals)
    - General documents without standard academic structure
    - Documents where section detection fails or is not needed
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.extract_metadata_from_text = self.config.get("extract_metadata_from_text", True)

    def extract_pdf(
        self,
        pdf_path: Path,
        phase_info: dict = None
    ) -> Tuple[Optional[List[ExtractedSection]], PDFMetadata]:
        """Extract PDF using simple full-text extraction.

        Returns:
            (None, metadata) - sections is None for generic extraction
        """
        logger.info(f"Generic extraction for: {pdf_path.name}")

        metadata = self._init_metadata(pdf_path, phase_info)
        metadata.document_type = "generic"
        metadata.extraction_method = "full_text"

        try:
            doc = fitz.open(pdf_path)
            metadata.total_pages = len(doc)

            # Extract metadata from PDF properties and first pages
            self._extract_pdf_metadata(doc, metadata)

            doc.close()
            return None, metadata

        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            metadata.extraction_method = "failed"
            return None, metadata

    def extract_full_text(self, pdf_path: Path) -> str:
        """Extract full text from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            max_pages = self.max_pages or len(doc)
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    cleaned = self._clean_text(text)
                    if cleaned.strip():
                        text_parts.append(cleaned)

            doc.close()
            full_text = "\n\n".join(text_parts)
            return self._final_cleanup(full_text)

        except Exception as e:
            logger.error(f"Error extracting full text from {pdf_path}: {e}")
            return ""

    def _extract_pdf_metadata(self, doc: fitz.Document, metadata: PDFMetadata):
        """Extract basic metadata from PDF properties and text."""
        # Extract from first pages
        first_pages_text = ""
        for page_num in range(min(3, len(doc))):
            first_pages_text += doc[page_num].get_text()

        # Try PDF properties first
        pdf_meta = doc.metadata
        if pdf_meta:
            if pdf_meta.get("title") and not metadata.title:
                title = pdf_meta["title"].strip()
                # Skip generic titles
                skip_patterns = ["Microsoft Word", "PowerPoint", ".docx", ".pdf", "Untitled"]
                if not any(skip.lower() in title.lower() for skip in skip_patterns) and len(title) > 5:
                    metadata.title = title

            # Year from dates
            for date_field in ["creationDate", "modDate"]:
                if pdf_meta.get(date_field) and not metadata.year:
                    year_match = re.search(r'(\d{4})', pdf_meta[date_field])
                    if year_match:
                        year = int(year_match.group(1))
                        if 1900 <= year <= datetime.now().year:
                            metadata.year = year
                            break

        # Extract title from text if not found
        if not metadata.title and first_pages_text:
            metadata.title = self._extract_title_from_text(first_pages_text)

        # Fallback to filename
        if not metadata.title:
            metadata.title = self._title_from_filename(metadata.filename)

        # DOI from text (often in header/abstract)
        if not metadata.doi and first_pages_text:
            metadata.doi = self._extract_doi_from_text(first_pages_text)

        # Authors from text
        if not metadata.authors and first_pages_text:
            metadata.authors = self._extract_authors_from_text(first_pages_text, metadata.title)

        # Year from text (fallback if not found via PDF properties)
        if not metadata.year and first_pages_text:
            metadata.year = self._extract_year_from_text(first_pages_text)

        # Try to extract summary/abstract
        if first_pages_text and not metadata.summary:
            # Look for executive summary or abstract
            summary_patterns = [
                r'(?i)executive\s+summary[\s:\-]*\n?(.{100,1500}?)(?=\n\s*(?:introduction|1\.|background))',
                r'(?i)abstract[\s:\-]*\n?(.{100,1500}?)(?=\n\s*(?:introduction|1\.))',
                r'(?i)overview[\s:\-]*\n?(.{100,1500}?)(?=\n\s*(?:introduction|1\.))',
            ]
            for pattern in summary_patterns:
                match = re.search(pattern, first_pages_text, re.DOTALL)
                if match:
                    summary = re.sub(r'\s+', ' ', match.group(1)).strip()
                    if len(summary) > 50:
                        metadata.summary = summary[:1500]
                        break

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from first lines of text."""
        lines = text.split('\n')

        for line in lines[:20]:
            line = line.strip()
            if len(line) < 8 or len(line) > 200:
                continue
            # Skip obvious non-titles
            line_lower = line.lower()
            skip = ['page', 'http', 'www.', '@', 'copyright', 'confidential']
            if any(s in line_lower for s in skip):
                continue
            # Skip if mostly numbers
            if sum(c.isdigit() for c in line) / max(len(line), 1) > 0.3:
                continue
            # Good candidate if starts with capital
            if line[0].isupper() and 15 <= len(line) <= 150:
                return re.sub(r'\s+', ' ', line)

        return None

    def _title_from_filename(self, filename: str) -> Optional[str]:
        """Generate title from filename."""
        stem = Path(filename).stem
        stem = re.sub(r'^[a-f0-9]{8}_', '', stem)  # Remove UUID prefix
        stem = re.sub(r'^\d{4}[_\-]', '', stem)    # Remove year prefix
        title = re.sub(r'[_\-]+', ' ', stem)
        title = re.sub(r'\s+', ' ', title).strip()
        if title:
            return title.title() if len(title) > 3 else None
        return None

    def _clean_text(self, text: str) -> str:
        """Remove noise from extracted text."""
        # Remove common PDF artifacts
        noise_patterns = [
            r'^\d+\s*$',  # Page numbers
            r'^Page\s+\d+',
            r'Downloaded from',
            r'©\s*\d{4}',
        ]
        cleaned = text
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)

        # Remove lines that are mostly non-alphanumeric
        lines = cleaned.split('\n')
        clean_lines = []
        for line in lines:
            if len(line.strip()) < 3:
                continue
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in line) / max(len(line), 1)
            if alnum_ratio >= 0.5:
                clean_lines.append(line)

        return '\n'.join(clean_lines)

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        # DOI pattern: 10.<digits>/<suffix>
        match = re.search(r'\b10\.\d{4,9}/[^\s"\'<>]+', text, re.IGNORECASE)
        if not match:
            return None
        doi = match.group(0).strip().rstrip(').,;')
        return doi.lower()

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract likely publication year from text."""
        current_year = datetime.now().year
        # Prefer copyright year
        copyright_match = re.search(r'©\s*(\d{4})', text)
        if copyright_match:
            year = int(copyright_match.group(1))
            if 1900 <= year <= current_year:
                return year

        # Fallback to first plausible year
        for year_match in re.finditer(r'\b(19|20)\d{2}\b', text):
            year = int(year_match.group(0))
            if 1900 <= year <= current_year:
                return year
        return None

    def _extract_authors_from_text(self, text: str, title: Optional[str]) -> Optional[List[str]]:
        """Extract author names from the first page text."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None

        # Identify title line index if possible
        title_idx = None
        if title:
            for i, line in enumerate(lines[:40]):
                if title.lower() in line.lower():
                    title_idx = i
                    break

        # Candidate lines: a few lines after title, or first 10 lines
        start = title_idx + 1 if title_idx is not None else 0
        candidate_lines = []
        for line in lines[start:start + 8]:
            if re.search(r'(?i)abstract|keywords|introduction|correspondence|received|accepted', line):
                break
            # Skip affiliations/emails
            if '@' in line or re.search(r'(?i)university|department|faculty|institute|school|centre|center', line):
                continue
            # Skip lines that are too long or too short
            if len(line) < 5 or len(line) > 120:
                continue
            candidate_lines.append(line)

        if not candidate_lines:
            # Try "By ..." pattern anywhere in first 20 lines
            for line in lines[:20]:
                by_match = re.match(r'(?i)^by\s+(.+)$', line)
                if by_match:
                    candidate_lines = [by_match.group(1).strip()]
                    break

        if not candidate_lines:
            return None

        # Combine and split into authors
        authors_text = " ".join(candidate_lines)
        authors_text = re.sub(r'\s+', ' ', authors_text)
        authors_text = re.sub(r'\d+', '', authors_text).strip()

        # Split on common delimiters
        raw_authors = re.split(r'\s*(?:,|;|&|and)\s*', authors_text, flags=re.IGNORECASE)
        authors = []
        for author in raw_authors:
            author = author.strip()
            # Require at least a first and last name-ish
            if len(author.split()) >= 2 and re.search(r'[A-Za-z]', author):
                authors.append(author)

        return authors or None

class BusinessExtractor(GenericExtractor):
    """Extractor optimized for business documents.

    Recognizes business document sections:
    - Executive Summary
    - Introduction/Background
    - Analysis/Findings
    - Recommendations
    - Conclusion
    - Appendix
    """

    BUSINESS_SECTIONS = {
        "executive_summary": r"(?i)^(executive\s+summary|summary)[\s:\-]",
        "introduction": r"(?i)^(introduction|background|overview)[\s:\-]",
        "analysis": r"(?i)^(analysis|findings|results|assessment)[\s:\-]",
        "recommendations": r"(?i)^(recommendations|proposed\s+actions|next\s+steps)[\s:\-]",
        "conclusion": r"(?i)^(conclusion|summary|closing)[\s:\-]",
        "appendix": r"(?i)^(appendix|appendices|attachments)[\s:\-]",
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.section_patterns = self.BUSINESS_SECTIONS
        self.section_confidence_threshold = self.config.get("section_confidence_threshold", 0.5)

    def extract_pdf(
        self,
        pdf_path: Path,
        phase_info: dict = None
    ) -> Tuple[Optional[List[ExtractedSection]], PDFMetadata]:
        """Extract business document with optional section detection."""
        logger.info(f"Business extraction for: {pdf_path.name}")

        metadata = self._init_metadata(pdf_path, phase_info)
        metadata.document_type = "business"

        try:
            doc = fitz.open(pdf_path)
            metadata.total_pages = len(doc)

            # Extract metadata
            self._extract_pdf_metadata(doc, metadata)

            # Try section detection
            sections, confidence = self._extract_with_sections(doc)

            if confidence >= self.section_confidence_threshold and sections:
                logger.info(f"Business section detection: {len(sections)} sections ({confidence:.1%} confidence)")
                metadata.sections_detected = len(sections)
                metadata.extraction_method = "section_aware"
                doc.close()
                return sections, metadata

            # Fallback to full-text
            logger.info(f"Using full-text extraction for {pdf_path.name}")
            metadata.extraction_method = "full_text"
            doc.close()
            return None, metadata

        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            metadata.extraction_method = "failed"
            return None, metadata

    def _extract_with_sections(self, doc: fitz.Document) -> Tuple[List[ExtractedSection], float]:
        """Detect and extract business document sections."""
        sections = []
        detected_types = set()

        # Extract all text with page tracking
        full_text = ""
        page_boundaries = [0]

        max_pages = self.max_pages or len(doc)
        for page_num in range(min(max_pages, len(doc))):
            page_text = doc[page_num].get_text()
            full_text += page_text + "\n\n"
            page_boundaries.append(len(full_text))

        lines = full_text.split('\n')
        current_section = None
        current_content = []
        current_start_pos = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for section matches
            detected_type = None
            for section_type, pattern in self.section_patterns.items():
                if re.match(pattern, line_stripped):
                    detected_type = section_type
                    break

            if detected_type:
                # Save previous section
                if current_section and current_content:
                    content_text = "\n".join(current_content).strip()
                    if len(content_text) > 50:
                        page_start, page_end = self._find_page_range(
                            current_start_pos, len('\n'.join(lines[:i])), page_boundaries
                        )
                        sections.append(ExtractedSection(
                            section_type=current_section,
                            content=content_text,
                            page_start=page_start,
                            page_end=page_end,
                            confidence=0.8
                        ))
                        detected_types.add(current_section)

                # Start new section
                current_section = detected_type
                current_content = [line]
                current_start_pos = len('\n'.join(lines[:i]))

            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            content_text = "\n".join(current_content).strip()
            if len(content_text) > 50:
                page_start, page_end = self._find_page_range(
                    current_start_pos, len(full_text), page_boundaries
                )
                sections.append(ExtractedSection(
                    section_type=current_section,
                    content=content_text,
                    page_start=page_start,
                    page_end=page_end,
                    confidence=0.8
                ))
                detected_types.add(current_section)

        # Calculate confidence based on expected sections
        expected = {"executive_summary", "introduction", "analysis", "conclusion"}
        detected_expected = expected.intersection(detected_types)
        confidence = len(detected_expected) / len(expected) if expected else 0

        return sections, confidence

    def _find_page_range(self, start_pos: int, end_pos: int, page_boundaries: List[int]) -> Tuple[int, int]:
        """Find page range for character positions."""
        page_start = 0
        page_end = 0

        for i, boundary in enumerate(page_boundaries[:-1]):
            next_boundary = page_boundaries[i + 1]
            if start_pos >= boundary and start_pos < next_boundary:
                page_start = i
            if end_pos >= boundary and end_pos < next_boundary:
                page_end = i
                break

        return page_start, max(page_start, page_end)


def create_extractor(
    extractor_type: str = "auto",
    config: dict = None,
    pdf_path: Path = None
) -> BaseExtractor:
    """Factory function to create appropriate extractor.

    Args:
        extractor_type: "academic", "business", "generic", or "auto"
        config: Extraction configuration
        pdf_path: Optional PDF path for auto-detection

    Returns:
        BaseExtractor instance
    """
    extractor_type = extractor_type.lower()

    if extractor_type == "academic":
        # Import here to avoid circular imports
        from ..pdf_extractor import AcademicPDFExtractor
        return AcademicPDFExtractor(config)

    elif extractor_type == "business":
        return BusinessExtractor(config)

    elif extractor_type == "generic":
        return GenericExtractor(config)

    elif extractor_type == "auto":
        # Auto-detect based on PDF content if path provided
        if pdf_path:
            detected_type = _detect_document_type(pdf_path)
            return create_extractor(detected_type, config)
        else:
            # Default to generic if no path
            return GenericExtractor(config)

    else:
        logger.warning(f"Unknown extractor type '{extractor_type}', using generic")
        return GenericExtractor(config)


def _detect_document_type(pdf_path: Path) -> str:
    """Auto-detect document type from PDF content.

    Analyzes first few pages for:
    - Academic indicators: abstract, references, DOI, methodology
    - Business indicators: executive summary, recommendations, action items

    Returns:
        "academic", "business", or "generic"
    """
    try:
        doc = fitz.open(pdf_path)
        first_pages_text = ""
        for page_num in range(min(5, len(doc))):
            first_pages_text += doc[page_num].get_text().lower()
        doc.close()

        # Academic indicators
        academic_patterns = [
            r'\babstract\b',
            r'\bmethodology\b',
            r'\breferences\b',
            r'\bdoi:\s*10\.',
            r'\bintroduction\b.*\bliterature\s+review\b',
            r'\bhypothesis\b',
            r'\bempirical\s+evidence\b',
        ]
        academic_score = sum(
            1 for p in academic_patterns
            if re.search(p, first_pages_text)
        )

        # Business indicators
        business_patterns = [
            r'\bexecutive\s+summary\b',
            r'\brecommendations?\b',
            r'\baction\s+items?\b',
            r'\bstakeholders?\b',
            r'\bROI\b',
            r'\bbudget\b',
            r'\bproject\s+plan\b',
            r'\bKPIs?\b',
        ]
        business_score = sum(
            1 for p in business_patterns
            if re.search(p, first_pages_text)
        )

        # Determine type
        if academic_score >= 3 and academic_score > business_score:
            logger.info(f"Auto-detected as academic (score: {academic_score})")
            return "academic"
        elif business_score >= 2 and business_score > academic_score:
            logger.info(f"Auto-detected as business (score: {business_score})")
            return "business"
        else:
            logger.info(f"Auto-detected as generic (academic: {academic_score}, business: {business_score})")
            return "generic"

    except Exception as e:
        logger.warning(f"Auto-detection failed: {e}, defaulting to generic")
        return "generic"

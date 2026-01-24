"""Academic PDF Extraction with Section Awareness

Extracts text and metadata from academic PDFs with intelligent section detection.
Falls back to full-text extraction if section detection confidence is low.

Adapted from personality RAG extraction patterns.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import fitz  # PyMuPDF
from datetime import datetime

logger = logging.getLogger(__name__)


# Section detection patterns for academic papers
SECTION_PATTERNS = {
    "abstract": r"(?i)^(abstract|summary)[\s:]",
    "introduction": r"(?i)^(1[\.\s]|introduction|background)[\s:]",
    "methods": r"(?i)^(\d+[\.\s]*)?(methodology|methods?|data|approach|research design)[\s:]",
    "results": r"(?i)^(\d+[\.\s]*)?(results?|findings?|analysis|empirical)[\s:]",
    "discussion": r"(?i)^(\d+[\.\s]*)?(discussion|implications?|interpretation)[\s:]",
    "conclusion": r"(?i)^(\d+[\.\s]*)?(conclusion|concluding|summary)[\s:]",
    "references": r"(?i)^(references|bibliography|works cited|literature)[\s:]"
}


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
    """Rich metadata extracted from academic PDF."""
    # Document identification
    doc_id: str
    filename: str
    file_path: str

    # Bibliographic
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None

    # Organization (from folder structure)
    phase: Optional[str] = None
    phase_name: Optional[str] = None
    topic_category: Optional[str] = None

    # Content
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None

    # Document stats
    total_pages: int = 0
    file_size_bytes: int = 0

    # Extraction stats
    sections_detected: int = 0
    extraction_method: str = "unknown"  # "section_aware" or "fixed_size"


class AcademicPDFExtractor:
    """Extract academic PDFs with section awareness."""

    def __init__(self, config: dict = None):
        """
        Initialize PDF extractor.

        Args:
            config: Extraction configuration from literature_config.yaml
        """
        self.config = config or {}
        self.section_patterns = SECTION_PATTERNS

        # Extraction settings
        self.use_section_detection = self.config.get("use_section_detection", True)
        self.section_confidence_threshold = self.config.get("section_confidence_threshold", 0.7)
        self.extract_sections = self.config.get("extract_sections", [
            "abstract", "introduction", "methods", "results", "discussion", "conclusion"
        ])
        self.skip_references = self.config.get("skip_references", True)
        self.max_pages = self.config.get("max_pages_per_pdf", None)
        self.metadata_pages = self.config.get("extract_first_n_pages_for_metadata", 3)

    def extract_pdf(self, pdf_path: Path, phase_info: dict = None) -> Tuple[Optional[List[ExtractedSection]], PDFMetadata]:
        """
        Extract PDF with section detection.

        Args:
            pdf_path: Path to PDF file
            phase_info: Dictionary with phase, phase_name, topic_category

        Returns:
            (sections, metadata) tuple
            - sections: List of ExtractedSection if section detection succeeds, None otherwise
            - metadata: PDFMetadata with extracted information
        """
        logger.info(f"Extracting PDF: {pdf_path.name}")

        # Initialize metadata
        metadata = self._init_metadata(pdf_path, phase_info)

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            metadata.total_pages = len(doc)

            # Extract metadata from PDF properties and first pages
            self._extract_pdf_metadata(doc, metadata)

            # Try section-aware extraction
            if self.use_section_detection:
                sections, confidence = self._extract_with_sections(doc)

                if confidence >= self.section_confidence_threshold:
                    logger.info(f"Section detection successful ({confidence:.1%} confidence)")
                    metadata.sections_detected = len(sections)
                    metadata.extraction_method = "section_aware"
                    if not metadata.abstract:
                        for section in sections:
                            if section.section_type == "abstract":
                                abstract_text = re.sub(r'\s+', ' ', section.content).strip()
                                if len(abstract_text) > 50:
                                    metadata.abstract = abstract_text[:2000]
                                break
                    doc.close()
                    return sections, metadata

            # Fallback to full-text extraction
            logger.info(f"Using full-text extraction for {pdf_path.name}")
            metadata.extraction_method = "full_text"
            doc.close()
            return None, metadata

        except Exception as e:
            logger.error(f"Error extracting {pdf_path}: {e}")
            metadata.extraction_method = "failed"
            return None, metadata

    def extract_full_text(self, pdf_path: Path) -> str:
        """Extract full text from PDF (fallback method)."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            max_pages = self.max_pages or len(doc)
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting full text from {pdf_path}: {e}")
            return ""

    def _init_metadata(self, pdf_path: Path, phase_info: dict = None) -> PDFMetadata:
        """Initialize metadata structure."""
        phase_info = phase_info or {}

        # Generate document ID
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
        # Use phase + sanitized filename
        phase_prefix = phase.lower().replace(" ", "_") if phase else "unknown"
        filename_part = pdf_path.stem[:50]  # Limit length
        filename_part = re.sub(r'[^a-z0-9_]', '_', filename_part.lower())
        return f"{phase_prefix}_{filename_part}"

    def _extract_pdf_metadata(self, doc: fitz.Document, metadata: PDFMetadata):
        """Extract metadata from PDF properties, first pages, and filename."""
        # First, try to extract from filename (often contains year and author)
        self._extract_from_filename(metadata)

        # Try PDF metadata properties
        pdf_meta = doc.metadata
        if pdf_meta:
            # Title (only if not a generic/system title)
            if pdf_meta.get("title"):
                title = pdf_meta["title"].strip()
                # Skip generic titles like "Microsoft Word - doc.docx"
                skip_patterns = ["Microsoft Word", "PowerPoint", ".docx", ".doc", ".pdf", "Untitled"]
                if not any(skip in title for skip in skip_patterns) and len(title) > 10:
                    metadata.title = title

            # Author(s)
            if pdf_meta.get("author") and not metadata.authors:
                author_str = pdf_meta["author"]
                # Skip system authors
                if author_str.lower() not in ["admin", "user", "unknown", "microsoft"]:
                    # Split by common separators
                    authors = re.split(r'[,;]|\s+and\s+', author_str)
                    metadata.authors = [a.strip() for a in authors if a.strip() and len(a.strip()) > 2]

            # Year (from creation date or modification date)
            if not metadata.year:
                for date_field in ["creationDate", "modDate"]:
                    if pdf_meta.get(date_field):
                        year_match = re.search(r'(\d{4})', pdf_meta[date_field])
                        if year_match:
                            year = int(year_match.group(1))
                            if 1950 <= year <= datetime.now().year:
                                metadata.year = year
                                break

        # Extract from first pages if metadata incomplete
        first_pages_text = ""
        for page_num in range(min(self.metadata_pages, len(doc))):
            first_pages_text += doc[page_num].get_text()

        # Extract title with improved heuristics
        if first_pages_text and not metadata.title:
            metadata.title = self._extract_title_from_text(first_pages_text)

        # Extract authors from text if still missing
        if first_pages_text and not metadata.authors:
            metadata.authors = self._extract_authors_from_text(first_pages_text)

        # Extract year from text (common patterns)
        if first_pages_text and not metadata.year:
            year_patterns = [
                r'(?:published|received|accepted).*?(\d{4})',  # Publication dates
                r'©\s*(\d{4})',  # Copyright year
                r'\b((?:19|20)\d{2})\b',  # Any year 1900-2099
            ]
            for pattern in year_patterns:
                year_match = re.search(pattern, first_pages_text, re.IGNORECASE)
                if year_match:
                    year = int(year_match.group(1))
                    if 1950 <= year <= datetime.now().year:  # Reasonable range
                        metadata.year = year
                        break

        # Extract DOI with multiple patterns
        if first_pages_text and not metadata.doi:
            doi_patterns = [
                r'doi[:\s]*(10\.\d{4,}/[^\s\]]+)',
                r'https?://doi\.org/(10\.\d{4,}/[^\s\]]+)',
                r'(10\.\d{4,}/[^\s\]]+)',
            ]
            for pattern in doi_patterns:
                doi_match = re.search(pattern, first_pages_text, re.IGNORECASE)
                if doi_match:
                    metadata.doi = doi_match.group(1).rstrip('.')
                    break

        # Extract abstract with improved patterns
        if first_pages_text and not metadata.abstract:
            abstract_patterns = [
                r'(?i)abstract[\s:\-]*\n?(.{100,2000}?)(?=\n\s*(?:keywords?|introduction|1\.|background))',
                r'(?i)abstract[\s:\-]*(.{100,2000}?)(?=\n\n\n)',
                r'(?i)summary[\s:\-]*\n?(.{100,2000}?)(?=\n\s*(?:keywords?|introduction|1\.))',
            ]
            for pattern in abstract_patterns:
                abstract_match = re.search(pattern, first_pages_text, re.DOTALL)
                if abstract_match:
                    abstract_text = abstract_match.group(1).strip()
                    abstract_text = re.sub(r'\s+', ' ', abstract_text)
                    if len(abstract_text) > 50:
                        metadata.abstract = abstract_text[:2000]
                        break

        # Extract keywords if present
        if first_pages_text:
            keywords_match = re.search(
                r'(?i)keywords?[\s:\-]*([^\n]+(?:\n[^\n]+)?)',
                first_pages_text
            )
            if keywords_match:
                kw_text = keywords_match.group(1)
                keywords = re.split(r'[,;•·]', kw_text)
                metadata.keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 2][:10]

    def _extract_from_filename(self, metadata: PDFMetadata):
        """Extract metadata from filename patterns like '2012_Thelen_Varieties.pdf'."""
        filename = metadata.filename
        stem = Path(filename).stem

        # Try to extract year from filename
        year_match = re.search(r'\b((?:19|20)\d{2})\b', stem)
        if year_match:
            year = int(year_match.group(1))
            if 1950 <= year <= datetime.now().year:
                metadata.year = year

        # Try to extract author from filename patterns like "2012_Thelen_" or "Thelen_2012"
        author_patterns = [
            r'^\d{4}[_\-\s]+([A-Z][a-z]+)',  # Year_Author
            r'^([A-Z][a-z]+)[_\-\s]+\d{4}',  # Author_Year
            r'^([A-Z][a-z]+(?:[_\-\s]+[A-Z][a-z]+)?)[_\-\s]',  # Author_Author_ at start
        ]
        for pattern in author_patterns:
            match = re.search(pattern, stem)
            if match:
                author_name = match.group(1).replace('_', ' ').replace('-', ' ')
                if len(author_name) > 2 and author_name.lower() not in ['the', 'and', 'for']:
                    metadata.authors = [author_name]
                    break

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title with improved heuristics."""
        lines = text.split('\n')

        # Skip common header elements
        skip_words = ['journal', 'volume', 'issue', 'doi', 'http', 'www', 'page',
                      'downloaded', 'accepted', 'received', 'published', 'copyright']

        # Look for title in first 20 lines
        for line in lines[:20]:
            line = line.strip()

            # Skip empty or very short lines
            if len(line) < 15:
                continue

            # Skip lines that look like headers/footers
            line_lower = line.lower()
            if any(skip in line_lower for skip in skip_words):
                continue

            # Skip lines that are all uppercase or have too many numbers
            if line.isupper() and len(line) > 50:
                continue
            if sum(c.isdigit() for c in line) > len(line) * 0.3:
                continue

            # Skip email addresses and URLs
            if '@' in line or 'http' in line.lower():
                continue

            # Looks like a reasonable title
            if 15 < len(line) < 300:
                # Clean up the title
                title = re.sub(r'\s+', ' ', line)
                return title

        return None

    def _extract_authors_from_text(self, text: str) -> Optional[List[str]]:
        """Extract authors from first pages text."""
        # Common author section patterns
        author_patterns = [
            # "by Author Name" or "By Author Name"
            r'(?i)\bby\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
            # "Author Name¹" (with superscript)
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)[\s]*[¹²³⁴⁵\*†‡]',
            # Multiple authors with "and"
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
        ]

        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:3000])  # Only check first 3000 chars
            for match in matches:
                if isinstance(match, tuple):
                    authors.extend(match)
                else:
                    authors.append(match)

        # Clean and deduplicate
        clean_authors = []
        seen = set()
        for author in authors:
            author = author.strip()
            # Skip if too short or looks like a word
            if len(author) < 4 or author.lower() in ['the', 'and', 'for', 'with']:
                continue
            if author.lower() not in seen:
                seen.add(author.lower())
                clean_authors.append(author)

        return clean_authors[:5] if clean_authors else None

    def _extract_with_sections(self, doc: fitz.Document) -> Tuple[List[ExtractedSection], float]:
        """
        Try to detect and extract sections.

        Returns:
            (sections, confidence) tuple
            - sections: List of ExtractedSection objects
            - confidence: Float between 0 and 1 indicating detection quality
        """
        sections = []
        detected_section_types = set()

        # Extract all text with page numbers
        full_text = ""
        page_boundaries = [0]  # Track character positions where pages start

        max_pages = self.max_pages or len(doc)
        for page_num in range(min(max_pages, len(doc))):
            page_text = doc[page_num].get_text()
            full_text += page_text + "\n\n"
            page_boundaries.append(len(full_text))

        # Try to detect sections
        lines = full_text.split('\n')
        current_section = None
        current_content = []
        current_start_pos = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Check if line matches any section pattern
            detected_type = None
            for section_type, pattern in self.section_patterns.items():
                if re.match(pattern, line_stripped):
                    # Skip references if configured
                    if section_type == "references" and self.skip_references:
                        continue

                    # Only extract configured sections
                    if section_type in self.extract_sections:
                        detected_type = section_type
                        break

            if detected_type:
                # Save previous section if exists
                if current_section and current_content:
                    content_text = "\n".join(current_content).strip()
                    if len(content_text) > 100:  # Minimum content length
                        # Find page range
                        page_start, page_end = self._find_page_range(
                            current_start_pos, len(full_text[:current_start_pos + len(content_text)]), page_boundaries
                        )

                        sections.append(ExtractedSection(
                            section_type=current_section,
                            content=content_text,
                            page_start=page_start,
                            page_end=page_end,
                            confidence=0.9  # High confidence for pattern-matched sections
                        ))
                        detected_section_types.add(current_section)

                # Start new section
                current_section = detected_type
                current_content = [line]
                current_start_pos = len('\n'.join(lines[:i]))

            elif current_section:
                # Continue current section
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            content_text = "\n".join(current_content).strip()
            if len(content_text) > 100:
                page_start, page_end = self._find_page_range(
                    current_start_pos, len(full_text), page_boundaries
                )
                sections.append(ExtractedSection(
                    section_type=current_section,
                    content=content_text,
                    page_start=page_start,
                    page_end=page_end,
                    confidence=0.9
                ))
                detected_section_types.add(current_section)

        # Calculate confidence based on detected sections
        expected_sections = {"abstract", "introduction", "methods", "results", "conclusion"}
        detected_expected = expected_sections.intersection(detected_section_types)
        confidence = len(detected_expected) / len(expected_sections)

        logger.debug(f"Detected {len(sections)} sections with {confidence:.1%} confidence")

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


def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Simple keyword extraction from text.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keyword strings
    """
    # Simple approach: extract capitalized phrases and common academic terms
    keywords = set()

    # Look for capitalized phrases (likely important concepts)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    keywords.update(capitalized[:max_keywords])

    # Look for common academic patterns
    academic_patterns = [
        r'(?i)(spatial panel|mixed methods|case study|varieties of capitalism)',
        r'(?i)(deindustrialization|tertiarization|entrepreneurship)',
        r'(?i)(regional\s+\w+)',
    ]

    for pattern in academic_patterns:
        matches = re.findall(pattern, text)
        keywords.update([m if isinstance(m, str) else m[0] for m in matches])

    return list(keywords)[:max_keywords]

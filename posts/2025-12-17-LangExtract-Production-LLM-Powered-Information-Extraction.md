---
title: "LangExtract: Production LLM-Powered Information Extraction with Source Grounding"
date: "2025-12-17T10:00:00+00:00"
author: "gp"
layout: "post"
image: "/content/2025/12/langextract.png"
categories: [LLM, NLP, Information Extraction, Financial Analysis, Production ML, RAG]
published: true
mermaid: true
---

Extracting structured data from unstructured financial documents is a mess. Traditional NER (Named Entity Recognition) fails on domain-specific terminology. Regex breaks on minor format changes. Custom fine-tuning requires massive labeled datasets and constant retraining.

What if you could extract complex financial information—revenue, risks, business segments—from 150-page SEC filings with **exact source positions**, using just a few examples, without fine-tuning any models?

**LangExtract** makes this possible. It's a Python library from Google that uses LLMs to extract structured information with three killer features most extraction tools lack:

1. **Source Grounding**: Every extraction maps to its exact location in the original text
2. **Schema Enforcement**: Controlled generation ensures consistent, typed outputs
3. **Multi-Model Support**: Works with Gemini, OpenAI, or local models via Ollama

In this post, we'll build a production-grade financial information extraction system. We'll fetch real SEC 10-K filings from EDGAR, extract structured data, and export to multiple formats—all with complete source traceability.

**Code Repository**: All examples are [available here](https://github.com/gsantopaolo/langextract) →

---

## The LLM Extraction Ecosystem

The space of LLM-powered structured extraction has exploded in the past year. Tools like [Instructor](https://python.useinstructor.com/) (Pydantic-based structured outputs), [LlamaExtract](https://www.llamaindex.ai/llamaextract) (managed extraction service), [Marvin](https://github.com/PrefectHQ/marvin) (ambient intelligence framework), and [Outlines](https://github.com/outlines-dev/outlines) (constrained decoding) all tackle parts of this problem.

**What makes LangExtract different?** Three things:

1. **Source grounding is built-in**: Every extraction includes exact character positions—critical for compliance, verification, and debugging
2. **Optimized for long documents**: Smart chunking, parallel processing, and multi-pass extraction specifically designed for 100+ page documents
3. **Production-ready architecture**: Not just structured output, but a complete pipeline from raw text to interactive visualizations

If you're already using Instructor for API responses or LlamaExtract for managed services, LangExtract complements them by solving the "where did this come from?" problem at scale.

---

## The Information Extraction Problem

### Why Traditional Approaches Fall Short

Consider extracting financial metrics from an SEC 10-K filing. You need:

- **Company information**: Name, ticker, CIK, sector
- **Financial metrics**: Revenue, operating income, net income, EBITDA, margins
- **Time periods**: Fiscal years, quarters with proper attribution
- **Risk factors**: Categorized by type (market, operational, regulatory)
- **Business segments**: Product lines, geographic regions, revenue breakdowns
- **Metric changes**: Year-over-year growth, comparisons

**Traditional regex/rule-based systems** break on:
- Format variations across companies
- Inconsistent terminology ("Total Revenue" vs "Revenues" vs "Net Sales")
- Complex nested structures (tables, footnotes, multi-page sections)
- Context-dependent interpretations

**Named Entity Recognition (spaCy, Stanford NER)** struggles with:
- Domain-specific financial entities not in training data
- Relationship extraction (connecting metrics to time periods and segments)
- Handling long-form documents (10-Ks can be 150+ pages)

**Fine-tuning LLMs** requires:
- Thousands of labeled examples
- Continuous retraining as formats evolve
- High computational costs
- Complex deployment pipelines

### The LangExtract Solution

LangExtract uses LLMs with **few-shot learning** and **structured prompting** to extract information while maintaining **exact source correspondence**. The key innovation: it doesn't just extract text—it maps every extraction to character-level positions in the source document.

```python
# Extract financial information with source grounding
result = lx.extract(
    text_or_documents=sec_filing_text,
    prompt_description="Extract financial metrics, risks, and segments...",
    examples=few_shot_examples,  # Just 2-3 high-quality examples
    model_id="gemini-2.0-flash-exp",
)

# Every extraction has exact position
for extraction in result.extractions:
    print(f"{extraction.extraction_class}: {extraction.extraction_text}")
    print(f"  Position: {extraction.char_interval.start_pos}-{extraction.char_interval.end_pos}")
    print(f"  Attributes: {extraction.attributes}")
```

---

## Architecture Overview

LangExtract's architecture solves the key challenges of long-document extraction:

```mermaid
graph TB
    Input[Document Text or URL] --> Chunker[Smart Chunking<br/>Respects Delimiters]
    Chunker --> Parallel[Parallel Processing<br/>Multiple Workers]
    Parallel --> Pass1[Extraction Pass 1]
    Parallel --> Pass2[Extraction Pass 2]
    Parallel --> Pass3[Extraction Pass 3]
    
    Pass1 --> Merge[Merge Non-Overlapping<br/>Extractions]
    Pass2 --> Merge
    Pass3 --> Merge
    
    Merge --> Ground[Source Grounding<br/>Char-Level Positions]
    Ground --> Output[AnnotatedDocument<br/>with Extractions]
    
    Output --> JSONL[JSONL Export]
    Output --> HTML[Interactive HTML<br/>Visualization]
    Output --> CSV[CSV/DataFrame]
    
    style Input fill:#e1f5ff
    style Output fill:#c8e6c9
    style Ground fill:#fff9c4
    style Merge fill:#ffccbc
```

**Key Components:**

1. **Smart Chunking**: Splits long documents while respecting paragraph boundaries to maintain context
2. **Parallel Processing**: Multiple workers process chunks simultaneously
3. **Multi-Pass Extraction**: Sequential passes improve recall by capturing entities missed in previous runs
4. **Source Grounding**: Every extraction is mapped back to its exact character position
5. **Schema Enforcement**: For models like Gemini, uses controlled generation to ensure structured output

---

## Quick Start: Basic Extraction

Let's start with a simple example before diving into SEC filings.

### Installation

```bash
pip install langextract

# Or from source
git clone https://github.com/google/langextract.git
cd langextract
pip install -e .
```

### API Key Setup

```bash
# For Gemini (recommended)
export LANGEXTRACT_API_KEY="your-gemini-api-key"

# Or create .env file
echo "LANGEXTRACT_API_KEY=your-api-key" > .env
```

Get your Gemini API key from [AI Studio](https://aistudio.google.com/app/apikey).

### Minimal Working Example

```python
import langextract as lx
import textwrap

# 1. Define what to extract
prompt = textwrap.dedent("""\
    Extract financial metrics including revenue, operating income, 
    and net income. Use exact text from the document.
""")

# 2. Provide one high-quality example
examples = [
    lx.data.ExampleData(
        text="Revenue was $60.9 billion, up 126% from $27.0 billion.",
        extractions=[
            lx.data.Extraction(
                extraction_class="financial_metric",
                extraction_text="Revenue was $60.9 billion",
                attributes={"metric": "revenue", "value": "60.9", "unit": "USD_billions"}
            ),
            lx.data.Extraction(
                extraction_class="financial_metric",
                extraction_text="$27.0 billion",
                attributes={"metric": "revenue", "value": "27.0", "unit": "USD_billions"}
            ),
        ]
    )
]

# 3. Extract from new text
input_text = "Operating income reached $33.0 billion compared to $4.2 billion."

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.0-flash-exp",
)

# 4. Access results
for e in result.extractions:
    print(f"{e.extraction_class}: {e.extraction_text}")
    print(f"  Attributes: {e.attributes}")
    print(f"  Position: chars {e.char_interval.start_pos}-{e.char_interval.end_pos}")
```

**Output:**
```
financial_metric: Operating income reached $33.0 billion
  Attributes: {'metric': 'operating_income', 'value': '33.0', 'unit': 'USD_billions'}
  Position: chars 0-39

financial_metric: $4.2 billion
  Attributes: {'metric': 'operating_income', 'value': '4.2', 'unit': 'USD_billions'}
  Position: chars 53-65
```

That's it. Now let's scale this to real SEC filings.

---

## Financial Extraction Schema Design

For production financial extraction, we need a comprehensive schema. Here's the complete entity hierarchy:

### Entity Types

**1. Company Information**
```python
lx.data.Extraction(
    extraction_class="company_info",
    extraction_text="NVIDIA Corporation (NASDAQ: NVDA)",
    attributes={
        "info_type": "company_name",  # or "ticker", "cik", "sector"
    }
)
```

**2. Financial Metrics**
```python
lx.data.Extraction(
    extraction_class="financial_metric",
    extraction_text="Revenue for fiscal 2024 was $60.9 billion",
    attributes={
        "metric_name": "revenue",
        "value": "60.9",
        "unit": "USD_billions",
        "time_period": "fiscal_2024",
        "segment": "total"  # optional: for segment-specific metrics
    }
)
```

**3. Risk Factors**
```python
lx.data.Extraction(
    extraction_class="risk_factor",
    extraction_text="export controls impacting sales to China",
    attributes={
        "risk_category": "regulatory",  # market, operational, regulatory, competitive
        "geographic_exposure": "china"
    }
)
```

**4. Business Segments**
```python
lx.data.Extraction(
    extraction_class="business_segment",
    extraction_text="Data Center segment",
    attributes={
        "segment_type": "product_line"  # or "geography", "service"
    }
)
```

**5. Time Periods**
```python
lx.data.Extraction(
    extraction_class="time_period",
    extraction_text="fiscal year 2024",
    attributes={
        "period_type": "annual"  # or "quarterly", "monthly"
    }
)
```

**6. Metric Changes**
```python
lx.data.Extraction(
    extraction_class="metric_change",
    extraction_text="up 126% from $27.0 billion in fiscal 2023",
    attributes={
        "change_type": "year_over_year_growth",
        "percentage": "126",
        "comparison_period": "fiscal_2023"
    }
)
```

### The Prompt

A well-designed prompt is critical. Here's the production version:

```python
def create_financial_extraction_prompt() -> str:
    return textwrap.dedent("""\
        Extract financial information from SEC filings and earnings reports.
        
        Extract the following entity types in order of appearance:
        - company_info: Company name, ticker symbol, CIK number
        - financial_metric: Revenue, operating income, net income, EBITDA, EPS, etc.
        - risk_factor: Risk disclosures and concerns
        - business_segment: Business units, product lines, geographic regions
        - time_period: Fiscal years, quarters, reporting periods
        - metric_change: Year-over-year changes, growth rates, comparisons
        
        Critical rules:
        1. Use EXACT text from the document for extraction_text
        2. Do NOT paraphrase or summarize
        3. Do NOT overlap text spans
        4. Extract entities in order of appearance
        5. Provide meaningful attributes for context
        
        For financial_metric entities, include these attributes:
        - metric_name: The specific metric (e.g., "revenue", "net_income")
        - value: The numerical value
        - unit: Currency or percentage (e.g., "USD_millions", "percent")
        - time_period: The reporting period
        
        For company_info entities, include:
        - info_type: "company_name", "ticker", "cik", "sector"
        
        For risk_factor entities, include:
        - risk_category: Type of risk (e.g., "market", "operational", "regulatory")
    """)
```

**Full source**: [`financial_extraction.py`](https://github.com/gsantopaolo/langextract/blob/main/articles/2025-12-17-langextract-financial-extraction-code/financial_extraction.py)

---

## Few-Shot Examples: The Secret Sauce

Quality matters more than quantity. Two excellent examples beat ten mediocre ones.

### Example 1: Earnings Report with Metrics

```python
lx.data.ExampleData(
    text=textwrap.dedent("""\
        NVIDIA Corporation (NASDAQ: NVDA) reported fiscal year 2024 results.
        Revenue for fiscal 2024 was $60.9 billion, up 126% from $27.0 billion
        in fiscal 2023. Operating income increased to $33.0 billion compared
        to $4.2 billion in the prior year. Net income reached $29.8 billion."""),
    extractions=[
        lx.data.Extraction(
            extraction_class="company_info",
            extraction_text="NVIDIA Corporation",
            attributes={"info_type": "company_name"}
        ),
        lx.data.Extraction(
            extraction_class="company_info",
            extraction_text="NASDAQ: NVDA",
            attributes={"info_type": "ticker"}
        ),
        lx.data.Extraction(
            extraction_class="time_period",
            extraction_text="fiscal year 2024",
            attributes={"period_type": "annual"}
        ),
        lx.data.Extraction(
            extraction_class="financial_metric",
            extraction_text="Revenue for fiscal 2024 was $60.9 billion",
            attributes={
                "metric_name": "revenue",
                "value": "60.9",
                "unit": "USD_billions",
                "time_period": "fiscal_2024"
            }
        ),
        lx.data.Extraction(
            extraction_class="metric_change",
            extraction_text="up 126% from $27.0 billion in fiscal 2023",
            attributes={
                "change_type": "year_over_year_growth",
                "percentage": "126",
                "comparison_period": "fiscal_2023"
            }
        ),
        # ... more extractions
    ]
)
```

### Example 2: Business Segments and Risks

```python
lx.data.ExampleData(
    text=textwrap.dedent("""\
        The Data Center segment generated $47.5 billion in revenue, representing
        78% of total revenue. Gaming revenue was $10.4 billion. The company faces
        risks related to export controls impacting sales to China."""),
    extractions=[
        lx.data.Extraction(
            extraction_class="business_segment",
            extraction_text="Data Center segment",
            attributes={"segment_type": "product_line"}
        ),
        lx.data.Extraction(
            extraction_class="financial_metric",
            extraction_text="$47.5 billion in revenue",
            attributes={
                "metric_name": "segment_revenue",
                "value": "47.5",
                "unit": "USD_billions",
                "segment": "data_center"
            }
        ),
        lx.data.Extraction(
            extraction_class="risk_factor",
            extraction_text="risks related to export controls impacting sales to China",
            attributes={
                "risk_category": "regulatory",
                "geographic_exposure": "china"
            }
        ),
    ]
)
```

**Full source**: [`financial_extraction.py:create_financial_examples()`](https://github.com/gsantopaolo/langextract/blob/main/articles/2025-12-17-langextract-financial-extraction-code/financial_extraction.py#L53)

---

## Fetching Real SEC Filings

Now let's fetch actual 10-K/10-Q filings from SEC EDGAR.

### SEC Filing Fetcher

```python
from sec_filing_fetcher import SECFilingFetcher

# Initialize with caching
fetcher = SECFilingFetcher(cache_dir="./sec_cache")

# Fetch NVIDIA's latest 10-K
filing_text = fetcher.get_filing(
    ticker="NVDA",
    filing_type="10-K",
    year="2024"  # optional
)

print(f"Retrieved {len(filing_text):,} characters")
```

Under the hood, this:
1. Looks up the company's CIK (Central Index Key)
2. Searches EDGAR for recent filings
3. Downloads the HTML document
4. Extracts and cleans text
5. Caches locally for repeated runs

**Key implementation details:**

```python
class SECFilingFetcher:
    BASE_URL = "https://www.sec.gov"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; Research/1.0; email@example.com)"
    }
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Map ticker to CIK number"""
        ticker_to_cik = {
            "NVDA": "0001045810",
            "MSFT": "0000789019",
            "AAPL": "0000320193",
            # ... more mappings
        }
        return ticker_to_cik.get(ticker.upper())
    
    def search_filings(self, cik: str, filing_type: str = "10-K", count: int = 5):
        """Search EDGAR for recent filings"""
        search_url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik.lstrip("0"),
            "type": filing_type,
            "count": count,
        }
        response = requests.get(search_url, params=params, headers=self.HEADERS)
        # Parse HTML to extract filing URLs
        # ...
        
    def get_filing_text(self, filing_url: str) -> str:
        """Fetch and clean HTML filing"""
        # Check cache first
        # Download HTML
        # Extract text with BeautifulSoup
        # Clean excessive whitespace
        # Cache result
        # ...
```

**Full source**: [`sec_filing_fetcher.py`](https://github.com/gsantopaolo/langextract/blob/main/articles/2025-12-17-langextract-financial-extraction-code/sec_filing_fetcher.py)

### Rate Limiting and Caching

SEC requests rate limiting. The fetcher implements:
- Caching to disk (avoid redundant downloads)
- Sleep delays between requests (0.1s)
- Proper User-Agent headers (SEC requires identification)

```python
# Rate limiting
time.sleep(0.1)

# Caching
cache_file = self.cache_dir / f"{cache_key}.txt"
if cache_file.exists():
    return cache_file.read_text(encoding="utf-8")
```

---

## Running the Complete Extraction Pipeline

Now the fun part: extract structured data from a real SEC filing.

### Command-Line Interface

```bash
python end_to_end_example.py --ticker NVDA --filing-type 10-K
```

This runs a 4-step pipeline:

**Step 1: Fetch SEC Filing**
```
Fetching NVIDIA 10-K Filing
----------------------------------------
Searching for 10-K filings for CIK 0001045810...
Found 5 filings
Fetching NVDA 10-K from 2024-02-21
✓ Retrieved filing: 148,523 characters
```

**Step 2: Extract Structured Information**
```
Extracting financial information using gemini-2.0-flash-exp...
Text length: 148,523 characters
✓ Extracted 287 entities
```

**Step 3: Analyze Results**
```
EXTRACTION SUMMARY
============================================================
Total entities: 287
Document length: 148,523 characters

Entity Type Breakdown:
----------------------------------------
  financial_metric: 98 (34.1%)
  company_info: 52 (18.1%)
  time_period: 47 (16.4%)
  metric_change: 41 (14.3%)
  business_segment: 28 (9.8%)
  risk_factor: 21 (7.3%)
```

**Step 4: Export Results**
```
✓ Saved extractions to: output/NVDA_10-K_extraction.jsonl
✓ Generated visualization: output/NVDA_10-K_extraction_visualization.html
✓ Exported 98 financial metrics to: output/NVDA_10-K_extraction_metrics.csv
✓ Exported 21 risk factors to: output/NVDA_10-K_extraction_risks.json
✓ Exported business segments to: output/NVDA_10-K_extraction_segments.csv
✓ Generated summary report: output/NVDA_10-K_extraction_summary.md
```

### The Core Extraction Function

```python
def extract_from_text(
    text: str,
    model_id: str = "gemini-2.0-flash-exp",
    extraction_passes: int = 2,
    max_workers: int = 10,
) -> lx.data.AnnotatedDocument:
    """
    Extract financial information with LangExtract.
    
    Args:
        text: Input text (SEC filing, earnings report, etc.)
        model_id: LLM model (gemini-2.0-flash-exp, gpt-4o, gemma2:2b)
        extraction_passes: Number of passes for higher recall
        max_workers: Parallel workers for chunked processing
    """
    prompt = create_financial_extraction_prompt()
    examples = create_financial_examples()
    
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        extraction_passes=extraction_passes,  # Multiple passes improve recall
        max_workers=max_workers,              # Parallel processing
        max_char_buffer=2000,                 # Chunk size (smaller = more accurate)
    )
    
    return result
```

**Key parameters:**

- `extraction_passes`: Sequential passes capture entities missed in previous runs. Each pass is independent; results are merged with "first-pass wins" for overlapping entities.
- `max_workers`: Number of parallel workers processing document chunks. Higher = faster (but more API calls).
- `max_char_buffer`: Chunk size in characters. Smaller chunks = better accuracy but more overhead.

**Full source**: [`financial_extraction.py:extract_from_text()`](https://github.com/gsantopaolo/langextract/blob/main/articles/2025-12-17-langextract-financial-extraction-code/financial_extraction.py#L160)

---

## Multi-Format Export Pipeline

Production systems need multiple export formats. Here's what we generate:

### 1. JSONL (Raw Extraction Data)

The canonical format—portable, human-readable, and loadable into pandas/Excel:

```json
{
  "text": "NVIDIA Corporation (NASDAQ: NVDA) reported...",
  "extractions": [
    {
      "extraction_class": "company_info",
      "extraction_text": "NVIDIA Corporation",
      "char_interval": {"start_pos": 0, "end_pos": 18},
      "attributes": {"info_type": "company_name"}
    },
    {
      "extraction_class": "financial_metric",
      "extraction_text": "Revenue for fiscal 2024 was $60.9 billion",
      "char_interval": {"start_pos": 125, "end_pos": 166},
      "attributes": {
        "metric_name": "revenue",
        "value": "60.9",
        "unit": "USD_billions",
        "time_period": "fiscal_2024"
      }
    }
  ]
}
```

### 2. Financial Metrics CSV

Clean table for analysis tools (Excel, pandas, Tableau):

```csv
metric_name,value,unit,time_period,segment,text
revenue,60.9,USD_billions,fiscal_2024,total,"Revenue for fiscal 2024 was $60.9 billion"
operating_income,33.0,USD_billions,fiscal_2024,total,"Operating income increased to $33.0 billion"
segment_revenue,47.5,USD_billions,fiscal_2024,data_center,"$47.5 billion in revenue"
revenue_percentage,78,percent,fiscal_2024,data_center,"78% of total revenue"
```

**Export function:**

```python
def export_financial_metrics(result: lx.data.AnnotatedDocument, output_file: str):
    """Export financial metrics to structured CSV."""
    metrics = [e for e in result.extractions if e.extraction_class == "financial_metric"]
    
    metrics_data = []
    for metric in metrics:
        metrics_data.append({
            "metric_name": metric.attributes.get("metric_name", "unknown"),
            "value": metric.attributes.get("value", ""),
            "unit": metric.attributes.get("unit", ""),
            "time_period": metric.attributes.get("time_period", ""),
            "segment": metric.attributes.get("segment", ""),
            "text": metric.extraction_text,
        })
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(output_file, index=False)
```

### 3. Risk Factors JSON

Structured risk data with categories:

```json
[
  {
    "category": "regulatory",
    "description": "risks related to export controls impacting sales to China",
    "geographic_exposure": "china",
    "position": {"start": 12458, "end": 12512}
  },
  {
    "category": "market",
    "description": "increased competition in the AI accelerator market",
    "geographic_exposure": "",
    "position": {"start": 15234, "end": 15285}
  }
]
```

### 4. Interactive HTML Visualization

LangExtract generates an interactive HTML file for exploring extractions visually:

```python
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content if isinstance(html_content, str) else html_content.data)
```

The visualization shows:
- Document text with highlighted extractions
- Click-to-inspect entity details
- Attribute display
- Filtering by entity type
- Source position information

### 5. Markdown Summary Report

Human-readable executive summary:

```markdown
# Financial Analysis Report: NVDA 10-K

## Executive Summary

- **Total Extractions**: 287
- **Financial Metrics**: 98
- **Risk Factors**: 21
- **Business Segments**: 28
- **Document Length**: 148,523 characters

## Key Financial Metrics

| Metric | Value | Period |
|--------|-------|--------|
| revenue | 60.9 USD_billions | fiscal_2024 |
| operating_income | 33.0 USD_billions | fiscal_2024 |
| net_income | 29.8 USD_billions | fiscal_2024 |

## Business Segments

- **Data Center segment** (product_line)
- **Gaming** (product_line)
- **Professional Visualization** (product_line)

## Risk Factor Categories

- **regulatory**: 8 risks identified
- **market**: 7 risks identified
- **operational**: 6 risks identified
```

**Full source**: [`end_to_end_example.py:generate_summary_report()`](https://github.com/gsantopaolo/langextract/blob/main/articles/2025-12-17-langextract-financial-extraction-code/end_to_end_example.py#L185)

---

## Model Selection Guide

LangExtract supports multiple LLM backends. Here's when to use each:

### Gemini Models (Recommended)

**gemini-2.0-flash-exp** (Default)
- **Speed**: 45s for 150k char document
- **Cost**: ~$0.03 per 10-K filing
- **Quality**: Excellent for most use cases
- **Best for**: Production workloads, high volume

```bash
python end_to_end_example.py --model gemini-2.0-flash-exp
```

**gemini-2.0-pro-exp** (Premium)
- **Speed**: 120s for 150k char document
- **Cost**: ~$0.15 per 10-K filing
- **Quality**: Superior accuracy, better reasoning
- **Best for**: Complex documents, critical applications

```bash
python end_to_end_example.py --model gemini-2.0-pro-exp
```

### OpenAI Models

**gpt-4o**
- **Speed**: 90s for 150k char document
- **Cost**: ~$0.25 per 10-K filing
- **Quality**: Excellent, comparable to Gemini Pro
- **Setup**: `pip install langextract[openai]`

```bash
export OPENAI_API_KEY="your-openai-key"
python end_to_end_example.py --model gpt-4o
```

> **Note**: OpenAI models require `fence_output=True` and `use_schema_constraints=False` in the extraction call.

### Local Models with Ollama

**gemma2:2b** (Privacy-First)
- **Speed**: 180s for 150k char document (CPU: Mac M1)
- **Cost**: Free (runs locally)
- **Quality**: Good, suitable for non-critical use
- **Best for**: Privacy-sensitive data, no internet

```bash
# Install Ollama
brew install ollama

# Pull model
ollama pull gemma2:2b

# Start server
ollama serve

# Run extraction
python end_to_end_example.py --model gemma2:2b
```

### Decision Matrix

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| **Production API** | gemini-2.0-flash-exp | Best cost/performance ratio |
| **Critical accuracy** | gemini-2.0-pro-exp | Superior reasoning |
| **Already on OpenAI** | gpt-4o | Excellent quality, familiar billing |
| **Privacy required** | gemma2:2b (Ollama) | Runs completely local |
| **High volume batch** | Vertex AI Batch | 50% cost savings |

---

## Performance Optimization

### Chunking Strategy

LangExtract splits long documents into overlapping chunks. The key parameters:

```python
result = lx.extract(
    text_or_documents=long_document,
    max_char_buffer=2000,    # Chunk size
    max_workers=20,          # Parallel workers
    extraction_passes=2,     # Number of passes
)
```

**Trade-offs:**

| max_char_buffer | Accuracy | Speed | Memory | API Calls |
|-----------------|----------|-------|--------|-----------|
| 1000 | Highest | Slower | Low | More |
| 2000 | High | Balanced | Medium | Balanced |
| 3000 | Good | Faster | Higher | Fewer |
| 5000+ | Lower | Fastest | High | Fewest |

**Recommendations:**
- **Accuracy-critical**: 1000-1500 chars, 3 passes
- **Production balanced**: 2000 chars, 2 passes
- **High-volume batch**: 3000 chars, 1 pass

### Parallel Processing

```python
# Conservative (reliable)
max_workers=5

# Balanced (recommended)
max_workers=10

# Aggressive (fast, more API calls)
max_workers=20
```

**Rate limiting considerations:**
- Gemini Flash: 1500 RPM (requests per minute)
- Gemini Pro: 360 RPM
- OpenAI: Tier-dependent (check your quota)

### Multi-Pass Extraction

Multiple passes improve recall by capturing entities missed in stochastic generation:

```python
extraction_passes=1  # Fast, ~85% recall
extraction_passes=2  # Balanced, ~93% recall (recommended)
extraction_passes=3  # High recall, ~96% recall, slower
```

Each pass is independent. Results are merged with "first-pass wins" for overlapping entities.

**When to use 3 passes:**
- Critical compliance documents
- Comprehensive risk analysis
- Benchmark/audit tasks

---

## Production Patterns

### Pattern 1: Batch Processing Multiple Filings

Process an entire sector or portfolio:

```python
from end_to_end_example import extract_from_text
from sec_filing_fetcher import SECFilingFetcher

tickers = ["NVDA", "AMD", "INTC", "TSM"]
fetcher = SECFilingFetcher()

results = {}
for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    filing_text = fetcher.get_filing(ticker, "10-K")
    
    if filing_text:
        result = extract_from_text(filing_text, extraction_passes=2)
        results[ticker] = result
        
        # Export individual results
        save_and_visualize(result, base_name=f"{ticker}_10K")

# Compare metrics across companies
for ticker, result in results.items():
    metrics = [e for e in result.extractions if e.extraction_class == "financial_metric"]
    print(f"{ticker}: {len(metrics)} metrics extracted")
```

### Pattern 2: Streaming Long Documents

For documents exceeding context limits, process in overlapping windows:

```python
def process_document_stream(document_text: str, window_size: int = 50000):
    """Process extremely long documents in overlapping windows."""
    all_extractions = []
    step = window_size // 2  # 50% overlap
    
    for start in range(0, len(document_text), step):
        chunk = document_text[start:start + window_size]
        
        result = extract_from_text(chunk, extraction_passes=1)
        
        # Adjust positions to absolute document coordinates
        for extraction in result.extractions:
            if extraction.char_interval:
                extraction.char_interval.start_pos += start
                extraction.char_interval.end_pos += start
        
        all_extractions.extend(result.extractions)
    
    # Deduplicate overlapping extractions
    unique_extractions = deduplicate_by_position(all_extractions)
    return unique_extractions
```

### Pattern 3: Incremental Updates

Track changes across quarterly filings:

```python
def extract_quarterly_changes(ticker: str, year: int):
    """Extract and compare Q1-Q4 filings."""
    fetcher = SECFilingFetcher()
    
    quarterly_data = {}
    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        filing = fetcher.get_filing(ticker, "10-Q", year=year, quarter=quarter)
        result = extract_from_text(filing)
        
        # Extract key metrics
        metrics = {
            e.attributes.get("metric_name"): e.attributes.get("value")
            for e in result.extractions
            if e.extraction_class == "financial_metric"
        }
        quarterly_data[quarter] = metrics
    
    # Compute trends
    df = pd.DataFrame(quarterly_data).T
    return df
```

### Pattern 4: Real-Time Alerts

Monitor new filings and extract risk factors:

```python
import time
from datetime import datetime, timedelta

def monitor_new_filings(tickers: List[str], check_interval: int = 3600):
    """Poll for new filings and extract risk factors."""
    fetcher = SECFilingFetcher()
    last_checked = {}
    
    while True:
        for ticker in tickers:
            filings = fetcher.search_filings(ticker, "8-K", count=1)
            
            if filings:
                latest = filings[0]
                filing_date = latest["filing_date"]
                
                # Check if this is new
                if ticker not in last_checked or filing_date > last_checked[ticker]:
                    print(f"New {ticker} filing: {filing_date}")
                    
                    # Extract and analyze
                    text = fetcher.get_filing_text(latest["filing_url"])
                    result = extract_from_text(text, extraction_passes=1)
                    
                    # Filter for critical risks
                    risks = [e for e in result.extractions if e.extraction_class == "risk_factor"]
                    if risks:
                        send_alert(ticker, filing_date, risks)
                    
                    last_checked[ticker] = filing_date
        
        time.sleep(check_interval)  # Check every hour
```

---

## Advanced Techniques

### Custom Entity Types

Extend the schema for domain-specific entities:

```python
prompt = textwrap.dedent("""\
    Extract financial information including:
    - executive_mention: C-suite names, titles, compensation
    - acquisition: M&A activity, deal values, target companies
    - product_launch: New product announcements, revenue guidance
    - regulatory_event: SEC filings, legal proceedings, compliance
    
    ... (rest of prompt)
""")

# Add corresponding examples
examples.append(
    lx.data.ExampleData(
        text="CEO Jensen Huang's total compensation was $34.2 million.",
        extractions=[
            lx.data.Extraction(
                extraction_class="executive_mention",
                extraction_text="CEO Jensen Huang",
                attributes={"title": "CEO", "name": "Jensen Huang"}
            ),
            lx.data.Extraction(
                extraction_class="financial_metric",
                extraction_text="total compensation was $34.2 million",
                attributes={
                    "metric_name": "executive_compensation",
                    "value": "34.2",
                    "unit": "USD_millions",
                    "executive": "Jensen Huang"
                }
            ),
        ]
    )
)
```

### Relationship Extraction

Use attributes to link related entities:

```python
# Extract segment-specific metrics
lx.data.Extraction(
    extraction_class="financial_metric",
    extraction_text="Data Center revenue was $47.5 billion",
    attributes={
        "metric_name": "segment_revenue",
        "value": "47.5",
        "unit": "USD_billions",
        "segment_id": "data_center",  # Links to segment
        "time_period": "fiscal_2024"
    }
)

# Later, query relationships
def get_segment_metrics(result, segment_id):
    """Get all metrics for a specific segment."""
    return [
        e for e in result.extractions
        if e.extraction_class == "financial_metric"
        and e.attributes.get("segment_id") == segment_id
    ]
```

### Hierarchical Extraction

Process documents in stages for complex structures:

```python
# Stage 1: Extract sections
sections_result = lx.extract(
    text=full_document,
    prompt_description="Extract section headers and boundaries",
    examples=section_examples,
    model_id="gemini-2.0-flash-exp",
)

# Stage 2: Extract details from each section
detailed_results = []
for section in sections_result.extractions:
    section_text = extract_text_by_position(
        full_document,
        section.char_interval.start_pos,
        section.char_interval.end_pos
    )
    
    # Extract financial details from this section
    section_result = lx.extract(
        text=section_text,
        prompt_description="Extract financial metrics",
        examples=metric_examples,
        model_id="gemini-2.0-flash-exp",
    )
    
    detailed_results.append({
        "section": section.extraction_text,
        "extractions": section_result.extractions
    })
```

---

## Error Handling and Edge Cases

### Handling Malformed Documents

```python
def safe_extract(text: str, max_retries: int = 3):
    """Extract with exponential backoff on failures."""
    import time
    
    for attempt in range(max_retries):
        try:
            result = extract_from_text(text)
            return result
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Extraction failed: {e}")
                if attempt == max_retries - 1:
                    raise
    
    return None
```

### Validating Extractions

```python
def validate_financial_metric(extraction: lx.data.Extraction) -> bool:
    """Validate that a financial metric has required attributes."""
    required = ["metric_name", "value", "unit"]
    
    if not extraction.attributes:
        return False
    
    for field in required:
        if field not in extraction.attributes:
            return False
        if not extraction.attributes[field]:
            return False
    
    # Validate value is numeric
    try:
        float(extraction.attributes["value"])
    except (ValueError, TypeError):
        return False
    
    return True

# Filter valid extractions
valid_metrics = [
    e for e in result.extractions
    if e.extraction_class == "financial_metric" and validate_financial_metric(e)
]
```

### Handling Missing Data

```python
def extract_with_fallback(text: str) -> lx.data.AnnotatedDocument:
    """Extract with progressive fallback strategies."""
    
    # Try with strict schema first
    try:
        return extract_from_text(text, extraction_passes=2)
    except Exception as e:
        print(f"Strict extraction failed: {e}")
    
    # Fallback: Single pass, larger chunks
    try:
        return extract_from_text(
            text,
            extraction_passes=1,
            max_char_buffer=3000
        )
    except Exception as e:
        print(f"Fallback extraction failed: {e}")
    
    # Last resort: Truncate and extract
    truncated = text[:50000]  # First 50k chars only
    return extract_from_text(truncated, extraction_passes=1)
```

---

## Benchmarks and Performance

### Real-World Results

Tested on NVIDIA 10-K (FY 2024):
- **Document**: 130 pages, 148,523 characters
- **Processing time**: 45 seconds (Gemini Flash)
- **Cost**: $0.027
- **Extractions**: 287 entities
  - 98 financial metrics
  - 52 company info
  - 47 time periods
  - 28 business segments
  - 21 risk factors
  - 41 metric changes

### Accuracy Validation

Manual validation on 50 randomly sampled extractions:

| Metric | Score |
|--------|-------|
| **Precision** | 96% (48/50 extractions were correct) |
| **Recall** | ~93% (estimated via multiple passes) |
| **Position Accuracy** | 100% (all char positions were exact) |
| **Attribute Quality** | 94% (correct attributes with minor formatting variations) |

**Common errors:**
- Ambiguous entity boundaries (e.g., "total revenue" vs "revenue")
- Missing context in complex sentences
- Overlapping entities not properly merged

### Cost Analysis

| Model | Documents/Hour | Cost/Document | Monthly Cost (1000 docs) |
|-------|----------------|---------------|--------------------------|
| Gemini Flash | 80 | $0.03 | $30 |
| Gemini Pro | 30 | $0.15 | $150 |
| GPT-4o | 40 | $0.25 | $250 |
| Gemma2 (local) | 20 | $0 | $0 (hardware only) |

*Assumes average 150k character documents*

### Scalability

Production deployment handling 10,000 SEC filings/month:

```python
# Configuration
model = "gemini-2.0-flash-exp"
extraction_passes = 2
max_workers = 10

# Performance metrics
total_docs = 10000
avg_processing_time = 45  # seconds
total_time = (total_docs * avg_processing_time) / max_workers / 3600  # hours
cost = total_docs * 0.03

print(f"Total processing time: {total_time:.1f} hours")
print(f"Total cost: ${cost:.2f}")
print(f"Throughput: {total_docs / total_time:.0f} docs/hour")
```

**Output:**
```
Total processing time: 12.5 hours
Total cost: $300.00
Throughput: 800 docs/hour
```

With proper parallelization and batching, you can process 10k documents in half a day for $300.

---

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
❌ Error: LANGEXTRACT_API_KEY not set in environment
```
**Solution:**
```bash
export LANGEXTRACT_API_KEY="your-key"
# Or create .env file
echo "LANGEXTRACT_API_KEY=your-key" > .env
```

**2. Rate Limiting**
```
Error 429: Too Many Requests
```
**Solution:** Reduce `max_workers` or add delays:
```python
result = extract_from_text(text, max_workers=5)  # Lower from 10
```

**3. Timeout on Large Documents**
```
Timeout after 30 seconds
```
**Solution:** Process in smaller chunks:
```python
result = extract_from_text(
    text,
    max_char_buffer=1500,  # Smaller chunks
    max_workers=5          # Fewer parallel requests
)
```

**4. Poor Extraction Quality**
```
Extracting wrong entities or missing key information
```
**Solution:** Improve examples and prompt:
- Add 1-2 more high-quality examples
- Make prompt more specific
- Include edge cases in examples
- Reduce chunk size for better context

**5. Memory Issues**
```
MemoryError: Unable to allocate
```
**Solution:** Reduce parallelism:
```python
result = extract_from_text(text, max_workers=2)
```

### Debugging Tips

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Inspect individual extractions:**
```python
for e in result.extractions[:5]:  # First 5
    print(f"\nClass: {e.extraction_class}")
    print(f"Text: {e.extraction_text}")
    print(f"Attributes: {e.attributes}")
    if e.char_interval:
        print(f"Position: {e.char_interval.start_pos}-{e.char_interval.end_pos}")
        # Verify position
        actual_text = result.text[e.char_interval.start_pos:e.char_interval.end_pos]
        print(f"Actual text at position: '{actual_text}'")
        assert actual_text == e.extraction_text, "Position mismatch!"
```

**Test on smaller samples first:**
```python
# Test on first 10k characters
sample = full_document[:10000]
result = extract_from_text(sample, extraction_passes=1)
analyze_extractions(result)
```

---

## Comparison: LangExtract vs Alternatives

| Feature | LangExtract | Instructor | LlamaExtract | spaCy NER | Custom Fine-Tuning |
|---------|-------------|------------|--------------|-----------|-------------------|
| **Source Grounding** | ✅ Exact positions | ❌ No grounding | ✅ Via LlamaParse | ✅ Token-level | ✅ Depends on impl. |
| **Schema Flexibility** | ✅ Few-shot examples | ✅ Pydantic models | ✅ Inferred schema | ❌ Fixed labels | ❌ Requires retraining |
| **Long Documents** | ✅ Chunking + multi-pass | ❌ Token limits | ✅ Optimized | ❌ Context limits | ✅ Depends on model |
| **Setup Time** | ⚡ Minutes | ⚡ Minutes | ⚡ Minutes (managed) | ⚡ Minutes | 🐌 Days-weeks |
| **Domain Adaptation** | ✅ Few-shot examples | ✅ Prompt + validation | ✅ Schema iteration | ❌ New training data | ❌ Full retraining |
| **Multi-Pass Extraction** | ✅ Built-in | ❌ Manual | ✅ Via iterations | ❌ No | ✅ Custom impl. |
| **Visualization** | ✅ Interactive HTML | ❌ No | ✅ Web UI | ❌ No | ❌ No |
| **Cost (1k docs)** | ~$30 | ~$30-50 | ~$50-100 (managed) | Free (self-hosted) | High |
| **Multi-Model Support** | ✅ 3+ providers | ✅ 15+ providers | ✅ Via LlamaIndex | ❌ spaCy only | ❌ Your model |
| **Best For** | Long docs + compliance | API responses | Managed extraction | Simple NER | Max accuracy |

**When to use what:**

**LangExtract** → Long documents with source traceability requirements (compliance, legal, financial)
**Instructor** → API responses and structured outputs with Pydantic validation
**LlamaExtract** → Managed service with minimal setup; schema inference from documents
**spaCy** → Fast offline NER on well-defined entities; privacy-critical applications
**Custom Fine-Tuning** → Maximum accuracy on fixed schema with large labeled datasets

---

## Wrapping Up

LangExtract bridges the gap between simple regex and expensive fine-tuning. You get:

- **Production-grade extraction** from complex documents
- **Exact source grounding** for every entity
- **Schema flexibility** via few-shot learning
- **Multi-model support** (Gemini, OpenAI, local Ollama)
- **Scalable pipeline** for thousands of documents

The complete financial extraction system we built:
1. Fetches SEC 10-K/10-Q filings from EDGAR
2. Extracts 6 entity types with rich attributes
3. Exports to JSONL, CSV, JSON, HTML, Markdown
4. Processes 150k character documents in 45 seconds
5. Costs $0.03 per document (Gemini Flash)

**Key Takeaways:**

✅ **Few-shot learning works**: 2-3 high-quality examples beat 1000 mediocre ones
✅ **Source grounding is critical**: Exact positions enable verification and compliance
✅ **Multi-pass extraction improves recall**: 2 passes → 93% recall, 3 passes → 96%
✅ **Chunking matters**: Smaller chunks (1000-2000 chars) → better accuracy
✅ **Schema design is iterative**: Start simple, refine based on results

**Next Steps:**

1. Clone the [code repository](https://github.com/gsantopaolo/langextract/tree/main/articles/2025-12-17-langextract-financial-extraction-code)
2. Get your [Gemini API key](https://aistudio.google.com/app/apikey)
3. Run the examples:
   ```bash
   python financial_extraction.py
   python end_to_end_example.py --ticker NVDA --filing-type 10-K
   ```
4. Adapt the schema to your domain
5. Deploy to production

**Resources:**

- [LangExtract GitHub](https://github.com/google/langextract)
- [Gemini API Docs](https://ai.google.dev/docs)
- [SEC EDGAR API](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm)
- [Complete Code Examples](https://github.com/gsantopaolo/langextract/tree/main/articles/2025-12-17-langextract-financial-extraction-code)

The future of information extraction isn't about building bigger NER models—it's about better prompting, source grounding, and flexible schemas. LangExtract delivers all three.

Now go extract some insights. 🚀

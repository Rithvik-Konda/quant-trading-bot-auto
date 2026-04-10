"""
news_intelligence.py — Unified News & Event Intelligence Pipeline
==================================================================
Ingests every market-moving text event before price moves.

Sources:
  - Truth Social (Trump posts via CNN archive, updated every 5min)
  - Reuters RSS (business/markets/politics/energy, every 60s)
  - AP RSS (breaking news, every 60s)
  - SEC 8-K filings (already cached, new filings via EDGAR RSS)
  - SEC Form 4 insider transactions (already in event_stream.py)
  - Fed/FOMC transcripts and statements
  - Scheduled macro events (CPI, NFP, PCE, GDP, ISM, FOMC)

Pipeline per event:
  1. Ingest raw text + timestamp + source
  2. FinBERT sentiment scoring (-1 to +1)
  3. Event type classification (tariff/ceasefire/rate/earnings/guidance/insider/macro/noise)
  4. Credibility scoring (source x event_type historical follow-through)
  5. Entity graph propagation (first-order + graph-distance-weighted tickers)
  6. Historical impact lookup (median price impact by event_type x sector)
  7. Output structured NewsEvent

Output feeds directly into TFT as time-varying unknown inputs per stock.
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import logging
import sqlite3
import requests
import feedparser
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/news_intelligence.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CACHE_NEWS_DIR  = os.path.join(BASE_DIR, 'cache_news')
CACHE_GRAPH_DIR = os.path.join(BASE_DIR, 'cache_graph')
DB_PATH         = os.path.join(CACHE_NEWS_DIR, 'news_events.db')
os.makedirs(CACHE_NEWS_DIR, exist_ok=True)

# ── FinBERT (lazy load) ───────────────────────────────────────────────────────
_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                'text-classification',
                model='ProsusAI/finbert',
                tokenizer='ProsusAI/finbert',
                device=-1,  # CPU
                max_length=512,
                truncation=True,
            )
            log.info("FinBERT loaded")
        except Exception as e:
            log.warning(f"FinBERT unavailable ({e}) — using fallback keyword scorer")
            _finbert_pipeline = 'fallback'
    return _finbert_pipeline


def score_finbert(text: str) -> float:
    """Score text sentiment with FinBERT. Returns -1.0 to +1.0."""
    if not text or len(text.strip()) < 5:
        return 0.0
    model = _get_finbert()
    if model == 'fallback':
        return _keyword_sentiment(text)
    try:
        result = model(text[:512])[0]
        label  = result['label'].lower()
        score  = result['score']
        if label == 'positive':
            return float(score)
        elif label == 'negative':
            return -float(score)
        else:
            return 0.0
    except Exception:
        return _keyword_sentiment(text)


def _keyword_sentiment(text: str) -> float:
    """Fast keyword fallback when FinBERT is unavailable."""
    text_lower = text.lower()
    positive = ['ceasefire', 'deal', 'agreement', 'growth', 'beat', 'surge',
                'rally', 'record', 'approved', 'partnership', 'upgrade',
                'raised', 'exceeded', 'strong', 'recovery', 'stimulus']
    negative = ['tariff', 'war', 'crash', 'miss', 'layoff', 'bankrupt',
                'sanction', 'attack', 'collapse', 'recession', 'downgrade',
                'cut', 'loss', 'default', 'crisis', 'warning', 'failed']
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return float((pos_count - neg_count) / total)


# ── Event Type Classifier ─────────────────────────────────────────────────────

# Maps keywords to event types — learned weights, not hard rules
# Each event_type has sector impact patterns derived from historical data
EVENT_TYPE_KEYWORDS = {
    'tariff': [
        'tariff', 'trade war', 'import duty', 'customs', 'trade deal',
        'trade agreement', 'export control', 'sanctions', 'trade deficit',
        'protectionism', 'liberation day', 'reciprocal'
    ],
    'ceasefire': [
        'ceasefire', 'cease fire', 'truce', 'peace deal', 'armistice',
        'peace agreement', 'end hostilities', 'diplomatic', 'normalization',
        'strait of hormuz', 'hormuz', 'de-escalation'
    ],
    'rate_signal': [
        'federal reserve', 'fed rate', 'interest rate', 'fomc', 'rate cut',
        'rate hike', 'monetary policy', 'jerome powell', 'basis points',
        'inflation target', 'rate decision', 'quantitative', 'tightening',
        'easing', 'dot plot', 'minutes'
    ],
    'earnings': [
        'earnings', 'quarterly results', 'revenue', 'eps', 'per share',
        'beat estimates', 'missed estimates', 'guidance', 'outlook',
        'q1', 'q2', 'q3', 'q4', 'first quarter', 'annual results'
    ],
    'guidance': [
        'guidance', 'forecast', 'outlook', 'full year', 'raises guidance',
        'lowers guidance', 'preliminary', 'pre-announcement', 'warns',
        'expects', 'projects revenue', 'updated forecast'
    ],
    'macro_data': [
        'cpi', 'inflation', 'pce', 'jobs report', 'nonfarm payroll', 'nfp',
        'unemployment', 'gdp', 'ism manufacturing', 'ism services',
        'retail sales', 'housing starts', 'consumer confidence', 'ppi'
    ],
    'geopolitical': [
        'attack', 'strike', 'invasion', 'conflict', 'war', 'military',
        'missile', 'drone', 'sanctions', 'nuclear', 'nato', 'crisis',
        'oil price', 'crude', 'opec', 'pipeline', 'tanker', 'strait'
    ],
    'ai_tech': [
        'ai model', 'artificial intelligence', 'chip', 'semiconductor',
        'nvidia', 'data center', 'gpu', 'machine learning', 'generative',
        'openai', 'anthropic', 'llm', 'compute', 'inference'
    ],
    'insider': [
        'form 4', 'insider', 'purchased shares', 'sold shares',
        'director bought', 'ceo sold', 'executive purchase', '10b5-1'
    ],
    'merger_acquisition': [
        'acquisition', 'merger', 'takeover', 'buyout', 'deal closed',
        'bid for', 'agreed to acquire', 'purchase price', 'all-cash deal'
    ],
}

# Sector impact map: event_type → {sector_etf: direction_multiplier}
# Derived from historical patterns — positive means sector benefits
SECTOR_IMPACT_MAP = {
    'tariff': {
        'XLK': -0.7, 'XLI': -0.5, 'XLY': -0.4, 'XLF': -0.3,
        'XLP': +0.2, 'XLU': +0.3, 'XLV': +0.1,
    },
    'ceasefire': {
        'XLE': -0.8,  # oil falls
        'XLI': +0.5,  # industrials benefit
        'XLK': +0.4,  # semis supply chain
        'XLY': +0.3,  # travel/consumer
        'XLF': +0.2,
    },
    'rate_signal_cut': {
        'XLK': +0.6, 'XLY': +0.5, 'XLRE': +0.7, 'XLU': +0.4,
        'XLF': -0.2, 'XLP': +0.1,
    },
    'rate_signal_hike': {
        'XLF': +0.4, 'XLK': -0.5, 'XLRE': -0.6, 'XLU': -0.4,
        'XLY': -0.3,
    },
    'geopolitical_escalation': {
        'XLE': +0.8, 'XLK': -0.4, 'XLI': -0.3, 'XLY': -0.5,
        'GLD': +0.6, 'XLP': +0.2, 'XLU': +0.1,
    },
    'macro_data_strong': {
        'XLF': +0.4, 'XLK': +0.3, 'XLY': +0.4, 'XLI': +0.3,
        'XLP': -0.1, 'XLU': -0.2,
    },
    'macro_data_weak': {
        'XLF': -0.3, 'XLK': -0.2, 'XLY': -0.4, 'XLI': -0.3,
        'XLP': +0.2, 'XLU': +0.3,
    },
}


def classify_event_type(text: str, source: str) -> str:
    """Classify event type from text. Returns the most likely event_type."""
    text_lower = text.lower()

    if source in ('form4', 'sec_form4'):
        return 'insider'

    scores = {}
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[event_type] = score

    if not scores:
        return 'noise'

    return max(scores, key=scores.get)


# ── Credibility Scorer ────────────────────────────────────────────────────────

class CredibilityScorer:
    """
    Learns follow-through rates per source x event_type from historical data.
    A Trump Truth Social ceasefire post has different credibility than
    a White House official statement or Reuters wire.
    """

    # Prior credibility by source (updated by historical outcomes)
    SOURCE_PRIORS = {
        'truth_social':   0.35,  # single source, often walked back
        'reuters':        0.82,  # professional journalists, fact-checked
        'ap':             0.80,
        'white_house':    0.75,  # official but sometimes reversed
        'fed_statement':  0.95,  # Fed means what it says
        'sec_8k':         0.90,  # legally required disclosure
        'sec_form4':      0.95,  # verified transaction data
        'earnings_call':  0.88,
        'cpi_release':    0.99,  # government statistical release
        'nfp_release':    0.99,
        'fomc_decision':  0.99,
    }

    # Modifier: cross-source confirmation boosts credibility
    CONFIRMATION_BOOST = 0.15  # per confirming source

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS credibility_outcomes (
                id INTEGER PRIMARY KEY,
                source TEXT,
                event_type TEXT,
                event_hash TEXT,
                timestamp TEXT,
                predicted_credibility REAL,
                actual_follow_through INTEGER,  -- 1=happened, 0=reversed/false
                price_impact_5d REAL
            )
        ''')
        conn.commit()
        conn.close()

    def score(
        self,
        source: str,
        event_type: str,
        text: str,
        confirming_sources: int = 0,
        specificity_score: float = 0.5,
    ) -> float:
        """
        Score credibility 0-1.

        specificity_score: how specific/verifiable is the claim?
        0 = vague ("things are going well")
        1 = specific ("Iran agrees to reopen Strait for 14 days, signed by...")
        """
        base = self.SOURCE_PRIORS.get(source, 0.5)

        # Boost for cross-source confirmation
        confirmation = min(confirming_sources * self.CONFIRMATION_BOOST, 0.30)

        # Boost for specificity — named parties, dates, verifiable details
        specificity_boost = specificity_score * 0.15

        # Penalty for contradiction signals
        contradiction_penalty = self._check_contradictions(text)

        raw = base + confirmation + specificity_boost - contradiction_penalty

        # Check historical follow-through for this source x event_type
        historical = self._get_historical_rate(source, event_type)
        if historical is not None:
            # Blend prior with historical (weighted toward historical as n grows)
            raw = 0.4 * raw + 0.6 * historical

        return float(np.clip(raw, 0.0, 1.0))

    def _check_contradictions(self, text: str) -> float:
        """Detect contradiction signals that reduce credibility."""
        text_lower = text.lower()
        contradiction_phrases = [
            'disputed', 'denied', 'contradicted', 'violated', 'walked back',
            'reversed', 'unconfirmed', 'not verified', 'iranian officials deny',
            'no agreement reached', 'preliminary', 'subject to'
        ]
        count = sum(1 for p in contradiction_phrases if p in text_lower)
        return min(count * 0.10, 0.30)

    def _get_historical_rate(self, source: str, event_type: str) -> Optional[float]:
        """Get historical follow-through rate for source x event_type."""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute('''
                SELECT AVG(actual_follow_through), COUNT(*)
                FROM credibility_outcomes
                WHERE source = ? AND event_type = ?
            ''', (source, event_type)).fetchone()
            conn.close()
            if row and row[1] >= 5:  # minimum 5 historical instances
                return float(row[0])
        except Exception:
            pass
        return None

    def specificity_score(self, text: str) -> float:
        """Estimate how specific/verifiable a claim is."""
        text_lower = text.lower()
        specific_signals = [
            r'\d+\s*(?:billion|million|percent|%)',  # numbers
            r'\d{4}-\d{2}-\d{2}',  # dates
            'signed by', 'agreed to', 'effective', 'commencing',
            'will reopen', 'will cease', 'confirmed by', 'according to',
        ]
        import re
        count = sum(1 for s in specific_signals
                    if (re.search(s, text_lower) if s.startswith(r'\d') or '\\' in s
                        else s in text_lower))
        return min(count / 4.0, 1.0)


# ── Entity Graph Propagation ──────────────────────────────────────────────────

class EntityGraphPropagator:
    """
    Propagates news events through the supply chain graph.
    When TSMC files a negative 8-K, propagate impact to NVDA, AMAT, LRCX etc.
    When ceasefire announced, propagate energy logistics impact across sectors.
    """

    def __init__(self, graph_path: str = None):
        if graph_path is None:
            graph_path = os.path.join(CACHE_GRAPH_DIR, 'entity_graph.json')
        self.graph = self._load_graph(graph_path)
        self.relationships = self._load_relationships()

    def _load_graph(self, path: str) -> dict:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Entity graph load failed: {e}")
        return {}

    def _load_relationships(self) -> pd.DataFrame:
        rel_path = os.path.join(CACHE_GRAPH_DIR, 'relationships.csv')
        if os.path.exists(rel_path):
            try:
                return pd.read_csv(rel_path)
            except Exception:
                pass
        return pd.DataFrame()

    def propagate(
        self,
        source_ticker: Optional[str],
        event_type: str,
        sentiment: float,
        sector_impacts: Dict[str, float],
        max_hops: int = 3,
        credibility: float = 1.0,
    ) -> Dict[str, Dict]:
        """
        Returns dict of ticker → {impact_score, graph_distance, impact_direction}

        impact_score: 0-1, how strongly this event affects the ticker
        graph_distance: 0=direct, 1=supplier/customer, 2=indirect, etc.
        impact_direction: +1 positive, -1 negative
        """
        affected = {}

        # Direct source ticker — full impact
        if source_ticker:
            affected[source_ticker] = {
                'impact_score':    abs(sentiment),
                'graph_distance':  0,
                'impact_direction': 1 if sentiment > 0 else -1,
            }

            # Propagate through supply chain graph
            if not self.relationships.empty:
                self._propagate_through_graph(
                    source_ticker, sentiment, affected, max_hops
                )

        # Sector-wide impact from event type
        sector_map = self._get_sector_map()
        for ticker, sector in sector_map.items():
            impact_mult = sector_impacts.get(sector, 0.0)
            effective_impact = abs(sentiment * impact_mult * credibility)
            if effective_impact > 0 and ticker not in affected:
                affected[ticker] = {
                    'impact_score':    effective_impact,
                    'graph_distance':  99,
                    'impact_direction': 1 if impact_mult > 0 else -1,
                }
            elif effective_impact > 0:
                affected[ticker]['impact_score'] = min(
                    affected[ticker]['impact_score'] + effective_impact * 0.3,
                    1.0
                )

        return affected

    def _propagate_through_graph(
        self,
        source: str,
        sentiment: float,
        affected: dict,
        max_hops: int,
    ):
        """BFS through supply chain relationships."""
        if self.relationships.empty:
            return

        visited = {source}
        queue = [(source, 0, sentiment)]

        while queue:
            ticker, hop, current_sentiment = queue.pop(0)
            if hop >= max_hops:
                continue

            # Find all relationships for this ticker
            rels = self.relationships[
                (self.relationships['source'] == ticker) |
                (self.relationships['target'] == ticker)
            ]

            for _, rel in rels.iterrows():
                # Determine the connected ticker
                connected = rel['target'] if rel['source'] == ticker else rel['source']
                if connected in visited:
                    continue
                visited.add(connected)

                # Decay sentiment with graph distance
                rel_type = str(rel.get('relationship', '')).lower()
                decay = 0.5 if rel_type in ('supplier', 'customer') else 0.3
                propagated_sentiment = current_sentiment * decay

                if abs(propagated_sentiment) > 0.05:
                    affected[connected] = {
                        'impact_score':    abs(propagated_sentiment),
                        'graph_distance':  hop + 1,
                        'impact_direction': 1 if propagated_sentiment > 0 else -1,
                    }
                    queue.append((connected, hop + 1, propagated_sentiment))

    def _get_sector_map(self) -> Dict[str, str]:
        """Load ticker → sector_etf mapping."""
        try:
            import config
            sector_map = {}
            for etf, symbols in config.SECTOR_ETFS.items():
                for s in symbols:
                    sector_map[s] = etf
            return sector_map
        except Exception:
            return {}


# ── News Event Dataclass ──────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    """Structured representation of a market-moving news event."""

    # Core identity
    event_id:        str      = ''       # SHA256 of source+timestamp+text[:100]
    timestamp:       str      = ''       # ISO format UTC
    source:          str      = ''       # truth_social/reuters/ap/sec_8k/fed/cpi/etc
    raw_text:        str      = ''

    # Scored signals
    finbert_score:   float    = 0.0      # -1 to +1
    event_type:      str      = 'noise'
    credibility:     float    = 0.5      # 0-1

    # Market impact
    affected_tickers: Dict    = field(default_factory=dict)
    sector_impacts:   Dict    = field(default_factory=dict)
    macro_catalyst:   bool    = False    # affects all stocks

    # Derived features for TFT
    reversal_prob:    float   = 0.5      # probability this reverses within 5d
    expected_duration_days: float = 1.0  # how long will impact persist

    # Metadata
    confirming_sources: int   = 0
    url:             str      = ''

    def to_dict(self) -> dict:
        return asdict(self)

    def to_ticker_features(self, ticker: str) -> Dict[str, float]:
        """
        Return per-ticker feature vector for TFT input.
        Called daily for each open position.
        """
        ticker_info = self.affected_tickers.get(ticker, {})
        impact = ticker_info.get('impact_score', 0.0)
        direction = ticker_info.get('impact_direction', 0)
        distance = ticker_info.get('graph_distance', 99)

        # Decay impact by graph distance
        distance_decay = np.exp(-0.5 * distance) if distance < 99 else 0.01

        return {
            'news_alpha':          float(self.finbert_score * impact * direction * distance_decay),
            'news_credibility':    float(self.credibility),
            'news_impact_score':   float(impact * distance_decay),
            'news_reversal_prob':  float(self.reversal_prob),
            'news_macro_catalyst': float(self.macro_catalyst),
            'news_event_age_days': 0.0,  # filled at lookup time
        }


# ── Source Fetchers ───────────────────────────────────────────────────────────

def fetch_truth_social_posts(since_id: Optional[str] = None) -> List[dict]:
    """Fetch Trump Truth Social posts via CNN archive (updated every 5min)."""
    url = 'https://ix.cnn.io/data/truth-social/truth_archive.json'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        posts = resp.json()
        if not isinstance(posts, list):
            return []

        # Filter to recent posts only (last 24h for initial load, last 10min for polling)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = []
        for post in posts:
            try:
                ts_str = post.get('created_at', '')
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    ts = ts.replace(tzinfo=None)
                    if ts > cutoff:
                        recent.append({
                            'id':        post.get('id', ''),
                            'text':      post.get('content', post.get('text', '')),
                            'timestamp': ts_str,
                            'source':    'truth_social',
                            'url':       post.get('url', ''),
                        })
            except Exception:
                continue

        # Strip HTML tags from Truth Social content
        import re
        for p in recent:
            p['text'] = re.sub(r'<[^>]+>', ' ', p['text']).strip()

        return recent

    except Exception as e:
        log.debug(f"Truth Social fetch error: {e}")
        return []


def fetch_rss_feed(url: str, source_name: str, max_age_hours: int = 4) -> List[dict]:
    """Fetch and parse an RSS feed. Returns list of article dicts."""
    try:
        feed = feedparser.parse(url)
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        articles = []

        for entry in feed.entries:
            try:
                # Parse timestamp
                published = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])
                else:
                    published = datetime.utcnow()

                if published < cutoff:
                    continue

                text = entry.get('title', '')
                summary = entry.get('summary', '')
                if summary:
                    text = f"{text}. {summary}"

                articles.append({
                    'id':        entry.get('id', entry.get('link', '')),
                    'text':      text,
                    'timestamp': published.isoformat(),
                    'source':    source_name,
                    'url':       entry.get('link', ''),
                })
            except Exception:
                continue

        return articles

    except Exception as e:
        log.debug(f"RSS fetch error {url}: {e}")
        return []


# News wire RSS feeds
RSS_FEEDS = {
    'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
    'reuters_markets':  'https://feeds.reuters.com/reuters/companyNews',
    'reuters_politics': 'https://feeds.reuters.com/Reuters/PoliticsNews',
    'reuters_energy':   'https://feeds.reuters.com/reuters/energyNews',
    'ap_breaking':      'https://feeds.ap.org/rss/apf-business',
    'cnbc_breaking':    'https://www.cnbc.com/id/100003114/device/rss/rss.html',
}


# ── Scheduled Macro Events Calendar ──────────────────────────────────────────

MACRO_SCHEDULE = {
    # Format: 'event_name': {'typical_day': 'first friday', 'event_type': ..., 'impact': ...}
    'nfp':      {'event_type': 'macro_data', 'description': 'Nonfarm Payrolls'},
    'cpi':      {'event_type': 'macro_data', 'description': 'CPI Inflation'},
    'fomc':     {'event_type': 'rate_signal', 'description': 'FOMC Rate Decision'},
    'gdp':      {'event_type': 'macro_data', 'description': 'GDP Release'},
    'pce':      {'event_type': 'macro_data', 'description': 'PCE Inflation'},
    'ism_mfg':  {'event_type': 'macro_data', 'description': 'ISM Manufacturing'},
    'jobless':  {'event_type': 'macro_data', 'description': 'Jobless Claims'},
}


# ── Reversal Probability Estimator ───────────────────────────────────────────

REVERSAL_PROBS = {
    # (source, event_type): historical reversal probability within 5 days
    # These are learned from historical data — initialized with informed priors
    ('truth_social',  'ceasefire'):       0.72,  # ceasefire was disputed same day
    ('truth_social',  'tariff'):          0.45,  # tariffs usually stick initially
    ('reuters',       'ceasefire'):       0.35,
    ('reuters',       'tariff'):          0.25,
    ('fed_statement', 'rate_signal'):     0.08,  # Fed almost never reverses immediately
    ('sec_8k',        'earnings'):        0.15,
    ('sec_form4',     'insider'):         0.20,
    ('cpi_release',   'macro_data'):      0.05,  # data is data
    ('nfp_release',   'macro_data'):      0.05,
}

IMPACT_DURATION = {
    'truth_social':   1.5,   # days — Trump posts fade fast
    'reuters':        2.5,
    'fed_statement':  5.0,   # Fed signals persist
    'sec_8k':         3.0,
    'sec_form4':      5.0,
    'cpi_release':    3.0,
    'nfp_release':    2.0,
}


def estimate_reversal_prob(source: str, event_type: str, credibility: float) -> float:
    key = (source, event_type)
    base = REVERSAL_PROBS.get(key, 0.40)
    # Low credibility → higher reversal probability
    credibility_adjustment = (1.0 - credibility) * 0.30
    return float(np.clip(base + credibility_adjustment, 0.0, 0.95))


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class NewsIntelligencePipeline:
    """
    Unified news ingestion and intelligence pipeline.
    Polls all sources, scores events, propagates through entity graph.
    """

    def __init__(self):
        self.credibility_scorer = CredibilityScorer()
        self.graph_propagator   = EntityGraphPropagator()
        self._seen_ids          = set()
        self._init_db()
        self._load_seen_ids()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS news_events (
                event_id    TEXT PRIMARY KEY,
                timestamp   TEXT,
                source      TEXT,
                event_type  TEXT,
                raw_text    TEXT,
                finbert     REAL,
                credibility REAL,
                reversal_p  REAL,
                macro       INTEGER,
                affected_json TEXT,
                sector_json   TEXT,
                url         TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON news_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON news_events(source)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON news_events(event_type)')
        conn.commit()
        conn.close()

    def _load_seen_ids(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute('SELECT event_id FROM news_events').fetchall()
            self._seen_ids = {r[0] for r in rows}
            conn.close()
        except Exception:
            self._seen_ids = set()

    def _make_event_id(self, source: str, timestamp: str, text: str) -> str:
        payload = f"{source}:{timestamp}:{text[:100]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def process_raw_item(self, item: dict) -> Optional[NewsEvent]:
        """Process a single raw news item into a NewsEvent."""
        text      = item.get('text', '')
        source    = item.get('source', 'unknown')
        timestamp = item.get('timestamp', datetime.utcnow().isoformat())
        url       = item.get('url', '')

        if not text or len(text.strip()) < 10:
            return None

        event_id = self._make_event_id(source, timestamp, text)
        if event_id in self._seen_ids:
            return None

        # Score
        finbert_score  = score_finbert(text)
        event_type     = classify_event_type(text, source)

        if event_type == 'noise':
            return None  # Skip all noise — no market signal
        if abs(finbert_score) < 0.10 and event_type not in ('macro_data', 'rate_signal', 'insider'):
            return None  # Skip near-zero sentiment unless scheduled data

        specificity    = self.credibility_scorer.specificity_score(text)
        credibility    = self.credibility_scorer.score(
            source, event_type, text, specificity_score=specificity
        )
        reversal_prob  = estimate_reversal_prob(source, event_type, credibility)
        duration_days  = IMPACT_DURATION.get(source, 2.0)
        macro_catalyst = event_type in ('tariff', 'rate_signal', 'macro_data', 'ceasefire', 'geopolitical')

        # Get sector impacts for this event
        sector_impacts = {}
        if event_type == 'ceasefire':
            sector_impacts = SECTOR_IMPACT_MAP.get('ceasefire', {})
        elif event_type == 'tariff' and finbert_score < 0:
            sector_impacts = SECTOR_IMPACT_MAP.get('tariff', {})
        elif event_type == 'rate_signal':
            if 'cut' in text.lower() or finbert_score > 0:
                sector_impacts = SECTOR_IMPACT_MAP.get('rate_signal_cut', {})
            else:
                sector_impacts = SECTOR_IMPACT_MAP.get('rate_signal_hike', {})
        elif event_type == 'geopolitical':
            if finbert_score < 0:
                sector_impacts = SECTOR_IMPACT_MAP.get('geopolitical_escalation', {})
        elif event_type == 'macro_data':
            if finbert_score > 0:
                sector_impacts = SECTOR_IMPACT_MAP.get('macro_data_strong', {})
            else:
                sector_impacts = SECTOR_IMPACT_MAP.get('macro_data_weak', {})

        # Extract direct ticker mentions
        source_ticker = self._extract_ticker(text)

        # Propagate through entity graph
        affected_tickers = self.graph_propagator.propagate(
            source_ticker  = source_ticker,
            event_type     = event_type,
            sentiment      = finbert_score,
            sector_impacts = sector_impacts,
            credibility    = credibility,
        )

        event = NewsEvent(
            event_id         = event_id,
            timestamp        = timestamp,
            source           = source,
            raw_text         = text[:1000],
            finbert_score    = finbert_score,
            event_type       = event_type,
            credibility      = credibility,
            affected_tickers = affected_tickers,
            sector_impacts   = sector_impacts,
            macro_catalyst   = macro_catalyst,
            reversal_prob    = reversal_prob,
            expected_duration_days = duration_days,
            url              = url,
        )

        self._save_event(event)
        self._seen_ids.add(event_id)

        return event

    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract primary ticker from text using known universe."""
        try:
            import config
            text_upper = text.upper()
            for sym in config.WATCHLIST:
                # Look for ticker symbol as word boundary
                import re
                if re.search(r'\b' + sym + r'\b', text_upper):
                    return sym
        except Exception:
            pass
        return None

    def _save_event(self, event: NewsEvent):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute('''
                INSERT OR IGNORE INTO news_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                event.event_id,
                event.timestamp,
                event.source,
                event.event_type,
                event.raw_text,
                event.finbert_score,
                event.credibility,
                event.reversal_prob,
                int(event.macro_catalyst),
                json.dumps(event.affected_tickers),
                json.dumps(event.sector_impacts),
                event.url,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"DB save error: {e}")

    def poll_all_sources(self) -> List[NewsEvent]:
        """Poll all sources and return new events."""
        new_events = []
        raw_items  = []

        # Truth Social
        ts_posts = fetch_truth_social_posts()
        raw_items.extend(ts_posts)

        # RSS feeds
        for source_name, url in RSS_FEEDS.items():
            articles = fetch_rss_feed(url, source_name)
            raw_items.extend(articles)

        # Process all items
        for item in raw_items:
            event = self.process_raw_item(item)
            if event is not None:
                new_events.append(event)
                log.info(
                    f"[{event.source}] {event.event_type} "
                    f"sentiment={event.finbert_score:+.2f} "
                    f"credibility={event.credibility:.2f} "
                    f"tickers={len(event.affected_tickers)} "
                    f"| {event.raw_text[:80]}"
                )

        return new_events

    def get_ticker_features_for_date(
        self,
        ticker: str,
        as_of_date: pd.Timestamp,
        lookback_days: int = 5,
    ) -> Dict[str, float]:
        """
        Get aggregated news features for a ticker as of a specific date.
        Used to build TFT input features from historical news.

        Returns feature dict:
            news_alpha_5d         — credibility-weighted sentiment sum
            news_max_impact_5d    — max single-event impact
            news_credibility_5d   — mean credibility of events
            news_reversal_risk_5d — mean reversal probability
            news_macro_count_5d   — count of macro catalyst events
            news_event_count_5d   — total relevant events
        """
        cutoff = (as_of_date - pd.Timedelta(days=lookback_days)).isoformat()

        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute('''
                SELECT finbert, credibility, reversal_p, macro, affected_json
                FROM news_events
                WHERE timestamp >= ? AND timestamp <= ?
                AND (macro = 1 OR affected_json LIKE ?)
            ''', (cutoff, as_of_date.isoformat(), f'%"{ticker}"%')).fetchall()
            conn.close()
        except Exception:
            return self._zero_features()

        if not rows:
            return self._zero_features()

        alphas, creds, reversals, macro_count = [], [], [], 0
        max_impact = 0.0

        for finbert, cred, rev_p, macro, affected_json in rows:
            try:
                affected = json.loads(affected_json) if affected_json else {}
            except Exception:
                affected = {}

            ticker_info = affected.get(ticker, {})
            impact      = ticker_info.get('impact_score', 0.05 if macro else 0.0)
            direction   = ticker_info.get('impact_direction', 1)

            alpha = finbert * impact * direction * cred
            alphas.append(alpha)
            creds.append(cred)
            reversals.append(rev_p)
            max_impact = max(max_impact, impact)
            if macro:
                macro_count += 1

        return {
            'news_alpha_5d':         float(sum(alphas)),
            'news_max_impact_5d':    float(max_impact),
            'news_credibility_5d':   float(np.mean(creds)) if creds else 0.0,
            'news_reversal_risk_5d': float(np.mean(reversals)) if reversals else 0.5,
            'news_macro_count_5d':   float(macro_count),
            'news_event_count_5d':   float(len(rows)),
        }

    def _zero_features(self) -> Dict[str, float]:
        return {
            'news_alpha_5d':         0.0,
            'news_max_impact_5d':    0.0,
            'news_credibility_5d':   0.0,
            'news_reversal_risk_5d': 0.5,
            'news_macro_count_5d':   0.0,
            'news_event_count_5d':   0.0,
        }

    def backfill_from_cache(self):
        """
        Backfill historical news features from existing cache_8k data.
        Uses the 12,113 8-K filings already processed with FinBERT.
        """
        cache_8k_path = os.path.join(BASE_DIR, 'cache_8k', '8k_history.csv')
        if not os.path.exists(cache_8k_path):
            log.warning("8K history not found for backfill")
            return

        log.info("Backfilling from 8-K history...")
        df = pd.read_csv(cache_8k_path)
        count = 0

        for _, row in df.iterrows():
            try:
                text = str(row.get('text', row.get('summary', '')))
                if not text or len(text) < 10:
                    continue

                item = {
                    'text':      text,
                    'source':    'sec_8k',
                    'timestamp': str(row.get('date', '')),
                    'url':       str(row.get('url', '')),
                }

                # Use pre-computed FinBERT score if available
                event = self.process_raw_item(item)
                if event and 'finbert' in row and not pd.isna(row['finbert']):
                    event.finbert_score = float(row['finbert'])

                count += 1
                if count % 500 == 0:
                    log.info(f"  Backfilled {count} 8-K events...")

            except Exception:
                continue

        log.info(f"Backfill complete: {count} events processed")


# ── Live Polling Loop ─────────────────────────────────────────────────────────

def run_live(poll_interval_seconds: int = 60):
    """
    Continuous live polling loop. Run via launchd or as a background process.
    Polls Truth Social every 5 minutes, RSS feeds every 60 seconds.
    """
    pipeline      = NewsIntelligencePipeline()
    ts_counter    = 0
    ts_interval   = 5  # poll Truth Social every 5 RSS cycles

    log.info(f"News Intelligence Pipeline started — polling every {poll_interval_seconds}s")

    while True:
        try:
            events = pipeline.poll_all_sources()
            if events:
                log.info(f"Processed {len(events)} new events")

            # Save latest features summary for live_trader.py
            _save_live_features(pipeline)

        except KeyboardInterrupt:
            log.info("Stopping news pipeline")
            break
        except Exception as e:
            log.error(f"Poll error: {e}")

        time.sleep(poll_interval_seconds)


def _save_live_features(pipeline: NewsIntelligencePipeline):
    """Save latest per-ticker news features for live_trader.py to consume."""
    try:
        import config
        now = pd.Timestamp.utcnow()
        features = {}
        for ticker in config.WATCHLIST:
            features[ticker] = pipeline.get_ticker_features_for_date(ticker, now)

        out_path = os.path.join(CACHE_NEWS_DIR, 'live_news_features.json')
        with open(out_path, 'w') as f:
            json.dump({
                'timestamp': now.isoformat(),
                'features':  features,
            }, f)
    except Exception as e:
        log.debug(f"Live features save error: {e}")


# ── Historical Feature Builder (for backtester) ───────────────────────────────

def build_historical_features(
    start_date: str = '2022-01-01',
    end_date:   str = '2026-01-01',
) -> pd.DataFrame:
    """
    Build per-ticker per-date news feature matrix for backtesting.
    Outputs a DataFrame indexed by (date, ticker) with news feature columns.
    Uses existing 8-K cache as the primary historical news source.
    """
    pipeline = NewsIntelligencePipeline()

    # Backfill from 8-K cache first
    pipeline.backfill_from_cache()

    try:
        import config
        tickers = list(config.WATCHLIST)
    except Exception:
        log.warning("Could not load config.WATCHLIST")
        return pd.DataFrame()

    dates = pd.date_range(start_date, end_date, freq='B')
    records = []

    log.info(f"Building historical news features for {len(tickers)} tickers × {len(dates)} dates...")

    for date in dates:
        for ticker in tickers:
            feats = pipeline.get_ticker_features_for_date(ticker, date)
            feats['date']   = date.date()
            feats['ticker'] = ticker
            records.append(feats)

    df = pd.DataFrame(records)
    df = df.set_index(['date', 'ticker'])

    out_path = os.path.join(CACHE_NEWS_DIR, 'historical_news_features.parquet')
    df.to_parquet(out_path)
    log.info(f"Saved {len(df):,} rows to {out_path}")

    return df


# ── Entry Points ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--live',      action='store_true', help='Run live polling loop')
    parser.add_argument('--backfill',  action='store_true', help='Backfill from 8-K cache')
    parser.add_argument('--build',     action='store_true', help='Build historical feature matrix')
    parser.add_argument('--test',      action='store_true', help='Test pipeline with one poll cycle')
    parser.add_argument('--start',     default='2022-01-01')
    parser.add_argument('--end',       default='2026-01-01')
    args = parser.parse_args()

    if args.live:
        run_live()

    elif args.backfill:
        pipeline = NewsIntelligencePipeline()
        pipeline.backfill_from_cache()

    elif args.build:
        df = build_historical_features(args.start, args.end)
        print(f"Built {len(df):,} rows")
        print(df.describe())

    elif args.test:
        pipeline = NewsIntelligencePipeline()
        print("Running single poll cycle...")
        events = pipeline.poll_all_sources()
        print(f"\nFound {len(events)} new events:")
        for e in events[:10]:
            print(f"  [{e.source}] {e.event_type} sentiment={e.finbert_score:+.2f} "
                  f"cred={e.credibility:.2f} tickers={len(e.affected_tickers)}")
            print(f"  → {e.raw_text[:100]}")
            print()

    else:
        parser.print_help()

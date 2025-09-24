#!/usr/bin/env python3
"""
Integrated G2P Module - Direct misaki integration
Replaces HTTP-based G2P service with direct function calls
"""

from typing import List
from dataclasses import dataclass
from misaki import en, espeak
import logging

logger = logging.getLogger(__name__)

@dataclass
class G2PItem:
    text: str
    language: str

@dataclass
class G2PResult:
    phonemes: str
    text: str
    lang: str
    lang_name: str

class IntegratedG2P:
    """Direct G2P processor using misaki library"""
    
    def __init__(self):
        self.g2p_processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize G2P processors for supported languages"""
        
        # Language code mappings (from g2p_service.py)
        self.aliases = {
            'en-us': 'a',
            'en-gb': 'b', 
            'es': 'e',
        }
        
        self.lang_codes = {
            'a': 'American English',
            'b': 'British English',
            'e': 'es',
        }
        
        # Initialize English G2P (American)
        try:
            fallback_us = espeak.EspeakFallback(british=False)
        except Exception as e:
            logger.warning(f"EspeakFallback (US) not enabled: {e}")
            fallback_us = None
        
        self.g2p_processors['a'] = en.G2P(trf=False, british=False, fallback=fallback_us, unk='')
        
        # Initialize English G2P (British)
        try:
            fallback_gb = espeak.EspeakFallback(british=True)
        except Exception as e:
            logger.warning(f"EspeakFallback (GB) not enabled: {e}")
            fallback_gb = None
        
        self.g2p_processors['b'] = en.G2P(trf=False, british=True, fallback=fallback_gb, unk='')
        
        # Initialize espeak-ng language processors
        espeak_languages = {
            'e': 'es',  # Spanish
        }
        
        for code, espeak_lang in espeak_languages.items():
            try:
                self.g2p_processors[code] = espeak.EspeakG2P(language=espeak_lang)
                logger.info(f"Initialized {espeak_lang} G2P processor")
            except Exception as e:
                logger.warning(f"Failed to initialize {espeak_lang} G2P: {e}")
        
        logger.info(f"Total G2P processors initialized: {len(self.g2p_processors)}")
        logger.info(f"Available languages: {list(self.g2p_processors.keys())}")
        
        # Suppress phonemizer warnings
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
    
    def convert_single_item(self, text: str, lang: str) -> G2PResult:
        """Convert a single text item to phonemes"""
        # Normalize and resolve language code using aliases
        lang_code = lang.lower()
        lang_code = self.aliases.get(lang_code, lang_code)
        
        if lang_code not in self.g2p_processors:
            supported = list(self.g2p_processors.keys())
            raise ValueError(f"Unsupported language: {lang} -> {lang_code}. Supported: {supported}")
        
        # Get appropriate G2P processor
        g2p = self.g2p_processors[lang_code]
        
        # Convert text to phonemes based on processor type
        if lang_code in ['a', 'b']:
            # English variants return (graphemes, tokens)
            _, tokens = g2p(text)
            phonemes = ''.join((t.phonemes or '') + (' ' if t.whitespace else '') for t in tokens).strip()
        elif lang_code == 'j':
            # Japanese returns (phonemes, tokens)
            phonemes, _ = g2p(text)
        elif lang_code == 'z':
            # Chinese returns (phonemes, tokens)
            phonemes, _ = g2p(text)
        else:
            # espeak-ng languages (e, f, h, i, p) return (phonemes, graphemes)
            phonemes, _ = g2p(text)
        
        return G2PResult(
            phonemes=phonemes,
            text=text,
            lang=lang_code,
            lang_name=self.lang_codes.get(lang_code, lang_code)
        )
    
    def convert_batch(self, items: List[G2PItem]) -> List[G2PResult]:
        """Convert multiple text items to phonemes"""
        results = []
        
        for item in items:
            if not item.text:
                # Skip empty text items but include them in results with empty phonemes
                results.append(G2PResult(
                    phonemes="",
                    text=item.text,
                    lang=item.language.lower(),
                    lang_name="Unknown"
                ))
                continue
            
            try:
                result = self.convert_single_item(item.text, item.language)
                results.append(result)
                logger.debug(f"Converted '{item.text}' ({item.language}) -> '{result.phonemes}'")
            except Exception as e:
                logger.error(f"G2P conversion error for '{item.text}' ({item.language}): {e}")
                # Include failed items with empty phonemes
                results.append(G2PResult(
                    phonemes="",
                    text=item.text,
                    lang=item.language.lower(),
                    lang_name="Error"
                ))
        
        logger.info(f"Batch processed {len(items)} items, {len([r for r in results if r.phonemes])} successful")
        return results
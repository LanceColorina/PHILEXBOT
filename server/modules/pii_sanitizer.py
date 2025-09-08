# pii_sanitizer.py
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIISanitizer:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.mask_map = {}
        self.reverse_map = {}
        self.counters = {}

    def sanitize(self, text: str):
        """
        Mask PII in text and return masked text + mask map.
        """
        results = self.analyzer.analyze(text=text, language="en")

        

        def custom_replacement(entity_type, original_text):
            self.counters.setdefault(entity_type, 1)
            key = f"{entity_type}_{self.counters[entity_type]}"
            self.counters[entity_type] += 1
            self.mask_map[key] = original_text
            self.reverse_map[original_text] = key
            return f"[{key}]"

        masked = text
        for r in results:
            original = text[r.start:r.end]
            masked = masked.replace(original, custom_replacement(r.entity_type, original))

        return masked, self.mask_map

    def restore(self, masked_text: str):
        """
        Restore masked text using mask_map.
        """
        restored = masked_text
        for key, original in self.mask_map.items():
            restored = restored.replace(f"[{key}]", original)
        return restored
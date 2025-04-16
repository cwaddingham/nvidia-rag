from typing import Dict, Any

class Document:
    """Simple document class to replace Langchain dependency"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata 
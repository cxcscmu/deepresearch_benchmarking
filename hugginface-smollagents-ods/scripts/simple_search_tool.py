from typing import List, Dict, Optional
from smolagents import Tool


class StaticTextViewer:
    def __init__(self, docs: List[Dict[str, str]], viewport_size: int = 1024):
        self.docs = docs
        self.current_doc_idx = 0
        self.viewport_size = viewport_size
        self.viewport_start = 0

    @property
    def current_doc(self) -> Dict[str, str]:
        return self.docs[self.current_doc_idx]

    def get_viewport(self) -> str:
        text = self.current_doc.get("text", "")
        start = self.viewport_start
        end = min(start + self.viewport_size, len(text))
        return text[start:end]

    def page_down(self) -> str:
        self.viewport_start = min(
            self.viewport_start + self.viewport_size,
            max(len(self.current_doc.get("text", "")) - 1, 0),
        )
        return self.get_viewport()

    def page_up(self) -> str:
        self.viewport_start = max(self.viewport_start - self.viewport_size, 0)
        return self.get_viewport()

    def select_doc(self, idx: int) -> str:
        self.current_doc_idx = max(0, min(idx, len(self.docs) - 1))
        self.viewport_start = 0
        return self.get_viewport()

    def find(self, query: str) -> Optional[str]:
        text = self.current_doc.get("text", "")
        idx = text.lower().find(query.lower(), self.viewport_start)
        if idx == -1:
            idx = text.lower().find(query.lower(), 0)
        if idx == -1:
            return None
        self.viewport_start = idx
        return self.get_viewport()


class StaticSearchTool(Tool):
    name = "static_web_search"
    description = "Search a static corpus and return top documents for a query."
    inputs = {
        "query": {"type": "string", "description": "The search query"},
        "top_k": {
            "type": "integer",
            "description": "Number of top documents to retrieve",
            "default": 5, 
            "nullable": True
        },
    }
    output_type = "string"

    def __init__(self, search_function, viewer: StaticTextViewer):
        super().__init__()
        self.search_function = search_function
        self.viewer = viewer

    def forward(self, query: str, top_k: int = 5) -> str:
        results = self.search_function(query, top_k=top_k)
        if not results:
            return f"No documents found for query: '{query}'"
        self.viewer.docs = results
        self.viewer.select_doc(0)

        formatted = "\n\n".join(
            f"### [{i+1}] {doc['title'] or 'Untitled'}\n{doc['text'][:500]}..."
            for i, doc in enumerate(results)
        )
        return f"Search results for query '{query}':\n\n{formatted}"


class ViewportTool(Tool):
    name = "view_current_page"
    description = "Returns the current visible portion of the selected document."
    inputs = {}
    output_type = "string"

    def __init__(self, viewer: StaticTextViewer):
        super().__init__()
        self.viewer = viewer

    def forward(self) -> str:
        return self.viewer.get_viewport()


class SelectDocTool(Tool):
    name = "select_document"
    description = "Select one of the retrieved documents to inspect more closely."
    inputs = {
        "index": {"type": "integer", "description": "Index of the document to select (starting from 0)"}
    }
    output_type = "string"

    def __init__(self, viewer: StaticTextViewer):
        super().__init__()
        self.viewer = viewer

    def forward(self, index: int) -> str:
        return self.viewer.select_doc(index)


class PageDownTool(Tool):
    name = "page_down"
    description = "Scroll the document viewport down."
    inputs = {}
    output_type = "string"

    def __init__(self, viewer: StaticTextViewer):
        super().__init__()
        self.viewer = viewer

    def forward(self) -> str:
        return self.viewer.page_down()


class PageUpTool(Tool):
    name = "page_up"
    description = "Scroll the document viewport up."
    inputs = {}
    output_type = "string"

    def __init__(self, viewer: StaticTextViewer):
        super().__init__()
        self.viewer = viewer

    def forward(self) -> str:
        return self.viewer.page_up()


class FindInDocTool(Tool):
    name = "find_in_document"
    description = "Search for a phrase within the current document and jump to its location."
    inputs = {
        "query": {"type": "string", "description": "The string to search for."}
    }
    output_type = "string"

    def __init__(self, viewer: StaticTextViewer):
        super().__init__()
        self.viewer = viewer

    def forward(self, query: str) -> str:
        result = self.viewer.find(query)
        if result is None:
            return f"'{query}' was not found in the current document."
        return result

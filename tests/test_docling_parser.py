"""Docling parser page-range tests without loading the external models."""

import gc
import importlib
import os
import sys
import types
import unittest
import weakref
from unittest.mock import patch


class _Item:
    def __init__(self, label, text):
        self.label = label
        self.text = text


class _Document:
    def __init__(self, page_range, empty=False):
        self.items = [
            _Item("header", "repeated header"),
            _Item("text", "" if empty else f"pages {page_range[0]}-{page_range[1]}"),
        ]

    def iterate_items(self):
        return ((item, 0) for item in self.items)

    def export_to_markdown(self):
        return "\n".join(item.text for item in self.items if item.text)


class _Converter:
    last_instance = None
    empty = False

    def __init__(self, **kwargs):
        self.calls = []
        self.document_refs = []
        _Converter.last_instance = self

    def convert(self, path, page_range):
        self.calls.append((path, page_range))
        document = _Document(page_range, empty=self.empty)
        self.document_refs.append(weakref.ref(document))
        return types.SimpleNamespace(document=document)


class DoclingParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_modules = {
            name: sys.modules.get(name)
            for name in (
                "pymupdf",
                "docling",
                "docling.datamodel",
                "docling.datamodel.base_models",
                "docling.datamodel.pipeline_options",
                "docling.document_converter",
                "docling_core",
                "docling_core.types",
                "docling_core.types.doc",
            )
        }

        pymupdf = types.ModuleType("pymupdf")
        class _Pdf:
            page_count = 5

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        pymupdf.open = lambda path: _Pdf()

        docling = types.ModuleType("docling")
        docling.__path__ = []
        datamodel = types.ModuleType("docling.datamodel")
        datamodel.__path__ = []
        base_models = types.ModuleType("docling.datamodel.base_models")
        base_models.InputFormat = types.SimpleNamespace(PDF="pdf")
        pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
        pipeline_options.PdfPipelineOptions = type("PdfPipelineOptions", (), {})
        document_converter = types.ModuleType("docling.document_converter")
        document_converter.DocumentConverter = _Converter
        document_converter.PdfFormatOption = lambda **kwargs: kwargs

        docling_core = types.ModuleType("docling_core")
        docling_core.__path__ = []
        core_types = types.ModuleType("docling_core.types")
        core_types.__path__ = []
        core_doc = types.ModuleType("docling_core.types.doc")
        core_doc.DocItemLabel = types.SimpleNamespace(
            PAGE_HEADER="header", PAGE_FOOTER="footer"
        )

        sys.modules.update(
            {
                "pymupdf": pymupdf,
                "docling": docling,
                "docling.datamodel": datamodel,
                "docling.datamodel.base_models": base_models,
                "docling.datamodel.pipeline_options": pipeline_options,
                "docling.document_converter": document_converter,
                "docling_core": docling_core,
                "docling_core.types": core_types,
                "docling_core.types.doc": core_doc,
            }
        )
        cls.module = importlib.import_module("app.services.parsers.docling_parser")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app.services.parsers.docling_parser", None)
        for name, module in cls.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def setUp(self):
        _Converter.empty = False
        _Converter.last_instance = None

    def test_converts_sequential_page_ranges_and_releases_documents(self):
        parser = self.module.DoclingParser(page_batch_size=2)
        documents = parser.parse(b"pdf", "guide.pdf")
        converter = _Converter.last_instance

        self.assertEqual(
            [page_range for _, page_range in converter.calls],
            [(1, 2), (3, 4), (5, 5)],
        )
        self.assertEqual(documents[0].text, "pages 1-2\n\npages 3-4\n\npages 5-5")
        self.assertEqual(documents[0].metadata["page_count"], 5)
        self.assertEqual(documents[0].metadata["page_batch_size"], 2)
        self.assertFalse(os.path.exists(converter.calls[0][0]))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in converter.document_refs))

    def test_rejects_empty_combined_markdown_and_removes_temp_file(self):
        _Converter.empty = True
        parser = self.module.DoclingParser(page_batch_size=5)

        with self.assertRaisesRegex(ValueError, "텍스트를 추출할 수 없습니다"):
            parser.parse(b"pdf", "empty.pdf")

        converter = _Converter.last_instance
        self.assertEqual(len(converter.calls), 1)
        self.assertFalse(os.path.exists(converter.calls[0][0]))

    def test_rejects_invalid_page_batch_size(self):
        with self.assertRaisesRegex(ValueError, "1 이상"):
            self.module.DoclingParser(page_batch_size=0)


if __name__ == "__main__":
    unittest.main()
